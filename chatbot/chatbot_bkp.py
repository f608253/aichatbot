import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY = "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"

#Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking queries", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=100,
        chunk_overlap=70,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)

    # #Generating Embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #
    # #Creating vector store - FAISS
    # vector_store = FAISS.from_texts(chunks,embeddings)
    #
    # #Get user question
    # user_question = st.text_input("Type your question here")

    # #Do similarity search
    # if user_question:
    #     match = vector_store.similarity_search(user_question)
    #     st.write(match)
    #
    #     #Output results
    #     chain = load_qa_chain(llm, chain_type="stuff")

