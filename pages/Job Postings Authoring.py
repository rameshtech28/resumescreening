import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import OCIGenAI
from langchain_core.prompts import PromptTemplate

import oci
st.sidebar.page_link("streamlit_app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Job Postings Authoring.py", label="Job Postings Authoring", icon="💼")
st.sidebar.page_link("pages/Smart_ATS.py", label="Smart ATS", icon="📑")
st.sidebar.page_link("pages/Chatbot.py", label="Chatbot", icon="🤖")
st.sidebar.page_link("pages/Chatbot_FAQ.py", label="Help Desk", icon="📜")

# Load secrets from Streamlit
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

# OCI configuration
config = {
    "user": user,
    "fingerprint": fingerprint,
    "key_file": key_file,
    "tenancy": tenancy,
    "region": region
}

# Initialize the OCI client
def initialize_llm(temperature=0, top_p=0, top_k=0, max_tokens=2000):
    try:
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config)
        llm = OCIGenAI(
            model_id="cohere.command",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=COMPARTMENT_ID,
            model_kwargs={"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens},
            client=client
        )
    except Exception as e:
        st.error(f"Error initializing OCIGenAI: {e}")
        raise e

    return llm

llm = initialize_llm()

# Example function to generate job postings
def generate_job_posting(job_title, responsibilities, qualifications, success_criteria):
    input_text = f"""
    Job Title: {job_title}
    Responsibilities: {responsibilities}
    Qualifications: {qualifications}
    Success Criteria: {success_criteria}
    """
    
    template = """
    Act like a skilled or very experienced ATS (Application Tracking System)
    with a deep understanding of the tech field, software engineering, data science, data analysis,
    and big data engineering. Your task is to generate the job description Create engaging job posting descriptions that convey a position’s requirements, qualifications, and success criteria.
    Here is the input: {input_text}
    """
    
    prompt = PromptTemplate.from_template(template)

    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.invoke({"input_text": input_text})
    
    return response

# Streamlit form for job postings
st.title("Job Postings Authoring")

job_title = st.text_input("Job Title")
responsibilities = st.text_area("Responsibilities")
qualifications = st.text_area("Qualifications")
success_criteria = st.text_area("Success Criteria")

if st.button("Generate Job Posting"):
    job_posting = generate_job_posting(job_title, responsibilities, qualifications, success_criteria)
    st.write("Generated Job Posting:")
    st.write(job_posting["text"])
