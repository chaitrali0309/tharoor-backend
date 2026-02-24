from dotenv import load_dotenv
load_dotenv()

import os
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1️⃣ ENV CHECK
# ==========================================
REQUIRED_ENV = ["GROQ_API_KEY", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_API_ENDPOINT"]
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")


# ==========================================
# 2️⃣ EMBEDDING — loaded ONCE at startup
# ==========================================
print("Loading embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)
print("Embedding model ready.")


# ==========================================
# 3️⃣ VECTOR STORES
# ==========================================
_astra_kwargs = dict(
    embedding=embedding,
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
)

vector_store_books      = AstraDBVectorStore(collection_name="books_collection",    **_astra_kwargs)
vector_store_parliament = AstraDBVectorStore(collection_name="parliament_clean_v1", **_astra_kwargs)
vector_store_profile    = AstraDBVectorStore(collection_name="profile_clean_v1",    **_astra_kwargs)

retriever_books      = vector_store_books.as_retriever(search_kwargs={"k": 6})
retriever_parliament = vector_store_parliament.as_retriever(search_kwargs={"k": 5})
retriever_profile    = vector_store_profile.as_retriever(search_kwargs={"k": 3})

print("AstraDB connected.")


# ==========================================
# 4️⃣ LLM
# ==========================================
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
)


# ==========================================
# 5️⃣ PROMPT — always attributes answers to Tharoor
# ==========================================
answer_prompt = ChatPromptTemplate.from_template("""
You are a research assistant specializing in Shashi Tharoor — Indian politician, author, and diplomat.

STRICT RULES:
- This assistant is ONLY about Shashi Tharoor. Always answer from HIS perspective.
- If the context contains quotes from OTHER people (Churchill, Modi, etc.), do NOT present those as Tharoor's views.
  Instead, you may say "Tharoor referenced [person] who said..." if relevant.
- Answer using ONLY the provided context.
- Be concise: 3-5 lines max.
- If "he", "his", "him" appears in the question, it always refers to Shashi Tharoor.
- Only say "Not available in indexed data." if context is truly unrelated to the question.

Recent conversation (for context):
{history}

Context from knowledge base:
{context}

Question: {question}
""")

chain = answer_prompt | llm | StrOutputParser()


# ==========================================
# 6️⃣ KEYWORD ROUTER — instant, no LLM call
# ==========================================
PROFILE_KEYWORDS = [
    "born", "birth", "family", "wife", "married", "children", "education",
    "school", "college", "early life", "career", "age", "personal", "son",
    "daughter", "mother", "father", "profile", "biography", "life"
]

PARLIAMENT_KEYWORDS = [
    "parliament", "speech", "debate", "lok sabha", "mp", "member",
    "constituency", "thiruvananthapuram", "vote", "bill", "session",
    "minister", "government", "policy", "question hour"
]

def smart_route(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in PROFILE_KEYWORDS):
        return "profile"
    if any(kw in q for kw in PARLIAMENT_KEYWORDS):
        return "parliament"
    return "books"


# ==========================================
# 7️⃣ DEDUPLICATE
# ==========================================
def _dedupe(docs):
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


# ==========================================
# 8️⃣ MAIN AGENT — accepts optional history
# ==========================================
def ask_agent(question: str, history: list = None) -> str:
    if not question or not question.strip():
        return "Please provide a question."

    question = question.strip()

    # Build history string for prompt
    history_str = ""
    if history:
        for turn in history[-4:]:   # last 4 turns max
            role = "User" if turn["role"] == "user" else "Assistant"
            history_str += f"{role}: {turn['text']}\n"

    route = smart_route(question)
    print(f"Route: {route} | Q: {question[:60]}")

    if route == "profile":
        docs = retriever_profile.invoke(question)
        if len(docs) < 2:
            docs += retriever_books.invoke(question)[:4]
    elif route == "parliament":
        docs = retriever_parliament.invoke(question)
    else:
        docs = retriever_books.invoke(question)
        if len(docs) < 4:
            docs += retriever_parliament.invoke(question)[:3]

    if not docs:
        return "No documents retrieved."

    docs = _dedupe(docs)[:8]
    context = "\n\n".join([d.page_content for d in docs])

    return chain.invoke({
        "context": context,
        "question": question,
        "history": history_str or "No prior conversation."
    })