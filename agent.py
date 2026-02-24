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


# Stronger retrieval window
retriever_books = vector_store_books.as_retriever(search_kwargs={"k": 12})
retriever_parliament = vector_store_parliament.as_retriever(search_kwargs={"k": 10})
retriever_profile = vector_store_profile.as_retriever(search_kwargs={"k": 6})


# ==========================================
# 4️⃣ LLM
# ==========================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# ==========================================
# 5️⃣ STRONG ANSWER PROMPT
# ==========================================

answer_prompt = ChatPromptTemplate.from_template("""
You are a strict political research assistant.

IMPORTANT RULES:
1. Answer ONLY using the provided context.
2. If the answer is not clearly found in the context, respond exactly with:
   Not available in indexed data.
3. Do NOT use external knowledge.
4. Do NOT guess.
5. Do NOT hallucinate.

Write in clear, professional language (3–6 lines).

Context:
{context}

Question:
{question}
""")


# ==========================================
# 6️⃣ MAIN AGENT FUNCTION
# ==========================================

def ask_agent(question: str) -> str:

    if not question or not question.strip():
        return "Please provide a question."

    question = question.strip()

    # 🔎 Retrieve from ALL collections (no router)
    docs = (
        retriever_profile.invoke(question)
        + retriever_books.invoke(question)
        + retriever_parliament.invoke(question)
    )

    if not docs:
        return "No documents retrieved."

    # 🧹 Deduplicate documents
    seen = set()
    unique_docs = []

    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    # Limit context size
    unique_docs = unique_docs[:12]

    context = "\n\n".join(
        [doc.page_content for doc in unique_docs]
    )

    chain = answer_prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": question
    })