from dotenv import load_dotenv
load_dotenv()

import os

from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1️⃣ REQUIRED ENV VARIABLES CHECK
# ==========================================

REQUIRED_ENV = [
    "GROQ_API_KEY",
    "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_API_ENDPOINT",
]

missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    raise RuntimeError(
        f"Missing environment variables: {missing}\n"
        f"Set them in Render Environment settings."
    )


# ==========================================
# 2️⃣ EMBEDDING MODEL
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 3️⃣ ASTRA VECTOR STORES
# ✅ FIX: Use the SAME collection names as in your notebook
# ==========================================

vector_store_books = AstraDBVectorStore(
    collection_name="books_collection",
    embedding=embedding,
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
)

vector_store_parliament = AstraDBVectorStore(
    collection_name="parliament_clean_v1",
    embedding=embedding,
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
)

vector_store_profile = AstraDBVectorStore(
    collection_name="profile_clean_v1",
    embedding=embedding,
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
)

retriever_books = vector_store_books.as_retriever(search_kwargs={"k": 8})
retriever_parliament = vector_store_parliament.as_retriever(search_kwargs={"k": 6})
retriever_profile = vector_store_profile.as_retriever(search_kwargs={"k": 4})


# ==========================================
# 4️⃣ LLM
# ==========================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ==========================================
# 5️⃣ ROUTER — decides which collection(s) to search
# ✅ FIX: Added back from notebook
# ==========================================

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


# ==========================================
# 6️⃣ QUERY REWRITER — converts question to search keywords
# ✅ FIX: Added back from notebook
# ==========================================

rewrite_prompt = ChatPromptTemplate.from_template("""
Convert the question into only keywords.
No quotes.
No explanation.
Return only keywords separated by spaces.

Question:
{question}
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# ==========================================
# 7️⃣ ANSWER PROMPT — more flexible, same as final notebook version
# ✅ FIX: Less strict, won't wrongly say "Not available"
# ==========================================

answer_prompt = ChatPromptTemplate.from_template("""
You are a political research assistant.

Task:
- Answer the question using ONLY the context.
- If the context includes a definition or explanation, summarize it clearly in 3–6 lines.
- If the context is about the topic but doesn't define it directly, explain what the context implies.
- Only say "Not available in indexed data." if the context is truly unrelated.

Write in simple, direct English.

Context:
{context}

Question:
{question}
""")


# ==========================================
# 8️⃣ HELPER — deduplicate while keeping order
# ==========================================

def _dedupe_keep_order(docs):
    seen = set()
    out = []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


# ==========================================
# 9️⃣ MAIN AGENT FUNCTION
# ==========================================

def ask_agent(question: str) -> str:

    if not question or not question.strip():
        return "Please provide a question."

    question = question.strip()

    # 🔎 Route to the right collection(s)
    route = router_chain.invoke({"question": question}).strip().lower()
    print("Route:", route)

    # 🔎 Rewrite question into search keywords
    optimized_query = rewrite_chain.invoke({"question": question}).strip()
    print("Query:", optimized_query)

    # 🔎 Retrieve from the appropriate collection(s)
    if route == "profile":
        docs = retriever_profile.invoke(optimized_query)

    elif route == "books":
        docs = retriever_books.invoke(optimized_query)

    elif route == "parliament":
        docs = retriever_parliament.invoke(optimized_query)

    elif route == "both":
        docs = (retriever_books.invoke(optimized_query)
                + retriever_parliament.invoke(optimized_query))

    else:  # all
        docs = (retriever_profile.invoke(optimized_query)
                + retriever_books.invoke(optimized_query)
                + retriever_parliament.invoke(optimized_query))

    if not docs:
        return "No documents retrieved."

    # 🧹 Deduplicate but keep order
    docs = _dedupe_keep_order(docs)

    # ✅ Keep only top chunks to avoid noise
    docs = docs[:10]

    # Build context with source info
    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source_type', '')}, Book: {d.metadata.get('book_name', '')}, Page: {d.metadata.get('page', '')}]\n{d.page_content}"
         for d in docs]
    )

    chain = answer_prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": question
    })
