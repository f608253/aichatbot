import os
import re
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

# Config

os.environ["TAVILY_API_KEY"] = "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"
SIMILARITY_THRESHOLD = 0.7

# Init models & tools

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatOllama(model="llama3:latest")
search_tool = TavilySearchResults()

# “deflection” phrases to detect vague answers

vague_phrases = [
    "does not", "cannot", "no information", "not mention",
    "don't know", "unable", "not possible", "don't" , "I'm not"
    "context appears", "unrelated", "irrelevant"
]

# Streamlit UI

st.set_page_config(page_title="Hybrid PDF→LLM→Web Chatbot", layout="wide")
st.header("📄 + 🧠 + 🌐 PDF → Local LLM → Web Fallback")

with st.sidebar:
    file = st.file_uploader("Upload a PDF file", type="pdf")

if not file:
    st.info("Please upload a PDF to get started.")
    st.stop()

# 1) Extract & chunk PDF

reader = PdfReader(file)
full_text = "".join(page.extract_text() or "" for page in reader.pages)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, length_function=len
)
chunks = splitter.split_text(full_text)
vector_store = FAISS.from_texts(chunks, embeddings)

# 2) User question

user_q = st.text_input("Ask a question:")
if not user_q:
    st.stop()

# ─────────────────────────────────────────────────────
# PDF retrieval + answer stage
# ─────────────────────────────────────────────────────

docs_and_scores = vector_store.similarity_search_with_score(user_q, k=3)
relevant = [
    (doc, score) for doc, score in docs_and_scores
    if score >= SIMILARITY_THRESHOLD
]

if relevant:
    context = "\n\n".join(doc.page_content for doc, _ in relevant)
    pdf_prompt = (
        "Use the context below to answer in one sentence:\n\n"
        f"{context}\n\nQ: {user_q}\nA:"
    )
    pdf_resp = llm.invoke(pdf_prompt)
    pdf_answer = pdf_resp.content.strip()

    is_pdf_vague = (
        any(p in pdf_answer.lower() for p in vague_phrases)
        or len(pdf_answer) < 40
    )
    if not is_pdf_vague:
        st.subheader("📄 Answer (from PDF)")
        st.write(pdf_answer)
        st.stop()

# ─────────────────────────────────────────────────────
# General‐knowledge LLM fallback
# ─────────────────────────────────────────────────────

gen_prompt = (
    "Answer the following question in one clear sentence "
    "using only your general knowledge:\n\n"
    f"Q: {user_q}\nA:"
)
gen_resp = llm.invoke(gen_prompt)
general_answer = gen_resp.content.strip()

is_gen_vague = (
    any(p in general_answer.lower() for p in vague_phrases)
    or len(general_answer) < 40
)
if not is_gen_vague:
    st.subheader("🧠 Answer (General knowledge)")
    st.write(general_answer)
    st.stop()

# ─────────────────────────────────────────────────────
# Final fallback: Web search + snippet summarization
# ─────────────────────────────────────────────────────

st.subheader("🌐 Answer (Web search)")

results = search_tool.run(user_q)

# Collect up to 5 “real” snippets (skip pure URLs or tiny text)

snippets = []
for hit in results[:5]:
    txt = hit["content"].strip()
    if re.match(r"^https?://", txt) or len(txt.split()) < 3:
        continue
    snippets.append(txt)

if snippets:
    combined = "\n\n".join(snippets)
    summarizer_prompt = (
        "Based on the following web search snippets, answer the question "
        "in one sentence:\n\n"
        f"{combined}\n\nQ: {user_q}\nA:"
    )
    summary_resp = llm.invoke(summarizer_prompt)
    web_answer = summary_resp.content.strip()
else:
    web_answer = "Sorry, I couldn’t scrape a concise answer from the web results."

st.write(web_answer)