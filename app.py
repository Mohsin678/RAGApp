import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["Hugging_face"] = os.getenv("Hugging_face")

embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")


st.title("Conversational Rag with pdf uploads and Chat history")
st.write("Upload pdfs and chat with their content")


api_key = st.text_input("Enter Your Groq Api key",type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key,model_name = "llama-3.1-8b-instant")

    session_id = st.text_input("session_id",value="Default session")

    if "store" not in st.session_state:
        st.session_state.store= {}

    
    uploaded_files = st.file_uploader("Choose a pdf file",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            tempdf = f"./temp.pdf"
        with open(tempdf,"wb")as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader  = PyPDFLoader(tempdf)
        docs = loader.load()
        documents.extend(docs)

        text_splitter  = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vdb = Chroma.from_documents(splits,embedding)
        retriever  = vdb.as_retriever()

        context_system_q_prompt = (
            """Given a chat history and the latest user question
                Which might reference context in the chat history
                formaulate a standalone question which can be understood,
                Without the chat history.Do not answer the question
                Just formulate it if need and otherwise return it asit
            """
        )
        context_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",context_system_q_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        history_aware_retriever  = create_history_aware_retriever(llm,retriever,context_q_prompt)


        system_prompt = (
            """    
            You are an assistant for question asnwering task
            use the following pieces of retrieved context to answer
            the question,If you dont know know the answer simply
            say you dont know,use three sentence maximum and keep answer
            concise.
            {context}
        """
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("enter your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag.invoke({"input":user_input},
                                                 config = {"configurable":{"session_id":session_id}})
            
            st.write(st.session_state.store)
            st.write("Assistant:",response["answer"])
            st.write("chat history:",session_history.messages)

    else:
        st.write("Please provide Groq key")