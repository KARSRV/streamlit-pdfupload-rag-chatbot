import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# set up streamlit page
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("Chat with your PDF")

# upload pdf file
pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf:
    # read text from pdf
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split text into small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings and store in vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # create chatbot with retrieval
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    # user input
    st.subheader("Ask a question about the PDF")
    query = st.text_input("Enter your question")

    # get answer
    if query:
        with st.spinner("Thinking"):
            response = qa.run(query)
        st.success(response)
