"""
Generator module — combines retrieved context with the question
and produces an answer.

By default uses a simple local template (no LLM / no API key needed).
Uncomment the OpenAI section to use GPT models instead.
"""

from langchain_core.documents import Document


# ── Option A: Local template (works with no API key) ─────────────────────────

def generate_answer(question: str, context_chunks: list[Document]) -> str:
    """
    Build a context string from retrieved chunks and format a simple answer.
    Useful for testing retrieval quality before wiring up a real LLM.
    """
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in context_chunks
    )

    return (
        f"Context retrieved:\n\n{context}\n\n"
        f"(Wire up an LLM in rag/generator.py to get a real answer to: '{question}')"
    )


# ── Option B: OpenAI LLM (requires OPENAI_API_KEY in .env) ───────────────────
#
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
#
# load_dotenv()
#
# def generate_answer(question: str, context_chunks: list[Document]) -> str:
#     context = "\n\n".join(doc.page_content for doc in context_chunks)
#
#     prompt = ChatPromptTemplate.from_template(
#         "Answer the question using only the context below.\n\n"
#         "Context:\n{context}\n\n"
#         "Question: {question}"
#     )
#
#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#     chain = prompt | llm
#
#     return chain.invoke({"context": context, "question": question}).content
