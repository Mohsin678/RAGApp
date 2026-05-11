import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_groq import ChatGroq

import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["Hugging_face"] = os.getenv("Hugging_face")

groq_api_key = os.getenv("Groq_key")

embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

llm = ChatGroq(groq_api_key = groq_api_key,model_name ="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """   
    Answer the questions based on the provided context only
    please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
"""
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("Research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = Chroma.from_documents(st.session_state.documents,st.session_state.embedding)

st.title("Rag Document QandA With groq and Llama3")

user_input = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Database is ready")

import time

if user_input:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input":user_input})
    st.write(response["answer"])


    with st.expander("Document Similarity search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----")