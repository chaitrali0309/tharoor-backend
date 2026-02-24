from dotenv import load_dotenv
load_dotenv()

import os
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================
# 1️⃣ CHECK REQUIRED ENV VARIABLES
# ==============================

REQUIRED_ENV = [
    "GROQ_API_KEY",
    "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_API_ENDPOINT",
]

missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    raise RuntimeError(
        f"Missing environment variables: {missing}\n"
        f"Set them before running server."
    )


# ==============================
# 2️⃣ ASTRA VECTOR STORES (NO LOCAL EMBEDDINGS)
# ==============================

vector_store_books = AstraDBVectorStore(
    collection_name="mp_shashi_tharoor_books_v1"
)

vector_store_parliament = AstraDBVectorStore(
    collection_name="mp_shashi_tharoor_parliament_v1"
)

vector_store_profile = AstraDBVectorStore(
    collection_name="mp_shashi_tharoor_profile_v1"
)

retriever_books = vector_store_books.as_retriever(search_kwargs={"k": 8})
retriever_parliament = vector_store_parliament.as_retriever(search_kwargs={"k": 6})
retriever_profile = vector_store_profile.as_retriever(search_kwargs={"k": 4})


# ==============================
# 3️⃣ LLM (GROQ)
# ==============================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ==============================
# 4️⃣ ROUTER PROMPT
# ==============================

router_prompt = ChatPromptTemplate.from_template("""
Decide which source should answer the question.
Respond with ONLY one word:

profile
books
parliament
both
all

Question:
{question}
""")

router_chain = router_prompt | llm | StrOutputParser()


# ==============================
# 5️⃣ QUERY REWRITER
# ==============================

rewrite_prompt = ChatPromptTemplate.from_template("""
Convert the question into only keywords.
No quotes.
No explanation.
Return only keywords separated by spaces.

Question:
{question}
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# ==============================
# 6️⃣ ANSWER PROMPT
# ==============================

answer_prompt = ChatPromptTemplate.from_template("""
You are a political research assistant.

- Answer using ONLY the provided context.
- Be clear and concise (3–6 lines).
- If unrelated, say: Not available in indexed data.

Context:
{context}

Question:
{question}
""")


# ==============================
# 7️⃣ MAIN FUNCTION
# ==============================

def ask_agent(question: str) -> str:

    if not question or not question.strip():
        return "Please provide a question."

    question = question.strip()

    # Step 1 — Route
    route = router_chain.invoke({"question": question}).strip().lower()

    # Step 2 — Rewrite
    optimized_query = rewrite_chain.invoke({"question": question}).strip()

    # Step 3 — Retrieve
    if route == "profile":
        docs = retriever_profile.invoke(optimized_query)

    elif route == "books":
        docs = retriever_books.invoke(optimized_query)

    elif route == "parliament":
        docs = retriever_parliament.invoke(optimized_query)

    elif route == "both":
        docs = (
            retriever_books.invoke(optimized_query)
            + retriever_parliament.invoke(optimized_query)
        )

    else:  # "all"
        docs = (
            retriever_profile.invoke(optimized_query)
            + retriever_books.invoke(optimized_query)
            + retriever_parliament.invoke(optimized_query)
        )

    if not docs:
        return "No documents retrieved."

    docs = docs[:10]

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    chain = answer_prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": question
    })