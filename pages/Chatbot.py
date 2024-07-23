import streamlit as st
import json
import time
import sqlite3
from datetime import date
from datetime import datetime, timedelta,timezone
import os
import base64
import PyPDF2 as pdf


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory.buffer import ConversationBufferMemory
# from langchain import PromptTemplate, FewShotPromptTemplate

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import oci

# Global Variables
CONFIG_PROFILE = st.secrets["CONFIG_PROFILE"] 
NAMESPACE = st.secrets["NAMESPACE"] 
BUCKET_NAME = st.secrets["BUCKET_NAME"] 
OBJECT_NAME = st.secrets["OBJECT_NAME"] 
COMPARTMENT_ID = st.secrets["COMPARTMENT_ID"] 
user = st.secrets["user"] 
fingerprint = st.secrets["fingerprint"] 
key_file = st.secrets["key_file"] 
tenancy = st.secrets["tenancy"] 
region = st.secrets["region"] 

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] =  st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

SESSION_ID = "abc12345"
DATABASE_NAME = "chat_history_table_session"

pdf_bucket_name = st.secrets["pdf_bucket_name"] #"demo_text_labeling"
history = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferMemory(chat_memory=history)
store = {}

#set config file
config = {
            "user":user,
            "fingerprint":fingerprint,       
            "key_file":key_file, 
            "tenancy":tenancy,        
            "region":region
        } 


st.sidebar.page_link("streamlit_app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Job Postings Authoring.py", label="Job Postings Authoring", icon="💼")
st.sidebar.page_link("pages/Smart_ATS.py", label="Smart ATS", icon="📑")
st.sidebar.page_link("pages/Chatbot.py", label="Chatbot", icon="🤖")
st.sidebar.page_link("pages/Chatbot_FAQ.py", label="Help Desk", icon="📜")

object_storage = oci.object_storage.ObjectStorageClient(config)

def initialize_llm(temperature=0.0,top_p=0,top_k=0,max_tokens=200):
    print(f"Temperature: {temperature}")
    print(f"Top_p: {top_p}")
    print(f"Top_k: {top_k}")
    print(f"Max_tokens: {max_tokens}")
    try:
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config)    
        
        llm =  OCIGenAI(
            model_id="cohere.command",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=COMPARTMENT_ID,
            model_kwargs={"temperature": 0.0, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens},     
            client=client
        )
        print("LLM initialized successfully")
    except Exception as e:
        print(f"Error initializing OCIGenAI: {e}")
        raise e

    return llm  


def create_vectorstore(docs):
    client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config)       
    
    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-english-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_ID,
        model_kwargs={"temperature": 0, "top_p": 0, "max_tokens": 512},
        client=client
    )
    return FAISS.from_documents(docs, embeddings)


def create_chains(llm, retriever):  

    template = """
            Act like a skilled or very experienced ATS (Application Tracking System)
            with a deep understanding of the tech field, software engineering, data science, data analysis,
            and big data engineering. Your task is to evaluate the resumes based on the given context.
            You should extarct the Name, skills, Projet details and Be able to locate a list of resumes matching certain keywords,
            qualifications, geographies, job duties, etc. that quickly helps narrow the search for candidates.
            Be able to match a list of resumes to job postings.
            You should compare the resumes and give best answer for provided context.
            the best assistance for selecting the resume. 
            If any  job description given in context then provide detailed comparision between resume 
            Answer the question based on the following context:
            {context}

            Question: {question}
            """
    
    prompt = PromptTemplate.from_template(template)

    chain = ( {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )   
    return chain

# def get_session_history(store, session_id):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def response_generator(response):
    lines = response.splitlines(keepends=True)
    for line in lines:
        for word in line.split():
            yield word + " "
            time.sleep(0.08)
        yield "\n"  # Yield a newline character to preserve line breaks

def display_chat_history(history):
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)


def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""

    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text()) + "\n"

    return text


# Connect to SQLite database
conn = sqlite3.connect(f'{DATABASE_NAME}.db')
c = conn.cursor()

def get_chatbot(llm): 
    try:
        loader = TextLoader("test2.json")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents) 
        # st.write("docs",docs)     

        vectorstore = create_vectorstore(docs)
        retriever = vectorstore.as_retriever()  

        rag_chain = create_chains(llm, retriever)
        st.toast('Embbedding done!', icon='🎉')
        return  rag_chain
        
    except Exception as e:
        st.error('Exception', e)
    
    # Create a table to store chat messages if not exists   


def main():
    # Define session history and other tabs
    tabs = ["Chatbot", "Session History"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    # Display content based on selected tab
    if selected_tab == "Chatbot":
        # Display current chat history
        st.subheader("Chatbot 🤖")
        c.execute(
        f'''CREATE TABLE IF NOT EXISTS {DATABASE_NAME} (session_id TEXT, AI_message TEXT, Human_message TEXT, date_val TEXT) ''')

        today = date.today()
        str_today = str(today)
        multiple_uploaded_files = st.sidebar.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True, help="Please upload the pdf files")
        submit = st.sidebar.button("Submit")  
        json_all = []
        count=0
        if submit:    
            if multiple_uploaded_files is not None:
                dict1 =  {}            
                for uploaded_file in multiple_uploaded_files:
                    
                    filename = uploaded_file.name
                    pdf_name_without_ext = filename.rstrip('.pdf')
                    print("pdf_name_without_ext",pdf_name_without_ext)
                    text = input_pdf_text(uploaded_file)
                    dict1 =  {
                                "unique_id":count+1,
                                "candidate name": pdf_name_without_ext,
                                "candidate resume": text,
                                "source file": filename
                            }
                    json_all.append(dict1)
                    count = count+1    

            with open('test2.json', 'w', newline='', encoding='utf-8') as json_file:
                json.dump(json_all, json_file, indent=4)

                st.toast('Json file created succefully for embbedding!', icon='🎉')

        with st.container():
            col1, col2, col3,col4 = st.columns(4)
            with col1:
                temperature  = st.slider("Tempreture:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            with col2:
                top_p = st.slider("Top_p:", min_value=0.00, max_value=1.00, value=0.00, step=0.01)
            with col3:
                max_tokens = st.slider("Max Tokens:", min_value=10, max_value=4000, value=512, step=1)    
            with col4:
                top_k = st.slider("Top_k:", min_value=0.00, max_value=1.00, value=0.00, step=0.01)

        display_chat_history(history)
                
        llm = initialize_llm(temperature,top_p,top_k,max_tokens)

        # Input from the user
        if prompt := st.chat_input():
            st.chat_message("human").write_stream(response_generator(prompt))
            

            # Invoke the conversational RAG chain
            rag_chain = get_chatbot(llm)
            response = rag_chain.invoke(prompt)
            # Update the history with the new human message
            history.add_user_message(prompt)

            # Update the history with the new AI response
            history.add_ai_message(response)

            # Display the AI response
            st.chat_message("ai").write_stream(response_generator(response))

            # Save chat message to the database
            c.execute(
                f"INSERT INTO {DATABASE_NAME} (session_id, AI_message, Human_message, date_val) VALUES (?,?,?,?)",
                (SESSION_ID, response,prompt, str_today))
            conn.commit()
            
            # get_chatbot()        

    elif selected_tab == "Session History":
        # Create a sidebar to display session history        
        # session_history = st.sidebar.expander("Session History")

        # Display chat history from the database
        st.write("Chat History:")
        unique_dates = c.execute(
            f"SELECT DISTINCT date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}'").fetchall()

        for date_value in unique_dates:
            with st.expander(date_value[0]):
                chat_history_date = c.execute(
                    f"SELECT AI_message, Human_message, date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}' and date_val='{date_value[0]}'"
                ).fetchall()

                for message_date in chat_history_date:
                    st.chat_message("human").write(message_date[1])
                    st.chat_message("ai").write(message_date[0])
                    st.markdown("<hr>", unsafe_allow_html=True)

        # all_data = c.execute(
        #             f"SELECT session_id,AI_message, Human_message, date_val FROM {DATABASE_NAME} where session_id='{SESSION_ID}'"
        #         ).fetchall()
        # st.write(all_data)
        # conn.close()

if __name__ == "__main__":
    main()