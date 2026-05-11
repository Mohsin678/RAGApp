import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("groq_key")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant please respond to user query"),
        ("user","{input}")
    ]
)


def generate_response(question,model,api_key,temperature,max_tokens):
    llm = ChatGroq(model=model,api_key=api_key)
    parser = StrOutputParser()
    chain = prompt|llm|parser
    answer = chain.invoke({"input":question})
    return answer

st.title("Enhanced Q&A chatbot with Groq")



llm = st.sidebar.selectbox("select open source model",["llama-3.1-8b-instant"])

temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


st.write("Go ahead and ask anything")
user_input = st.text_input("you:")



if user_input:
    response = generate_response(user_input,llm,api_key,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide  the user input")