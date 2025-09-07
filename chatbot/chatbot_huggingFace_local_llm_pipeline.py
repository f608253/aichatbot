import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

# 🧠 Load local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧠 Load local language model for QA
local_llm = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=local_llm)

# 🎯 Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("📄 Chat with Your PDF")

with st.sidebar:
    st.title("Upload Document")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # 🔍 Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 🧠 Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # 🔗 Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # 💬 User question
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        answer = qa_chain.run(user_question)
        st.subheader("🧠 Answer")
        st.write(answer)