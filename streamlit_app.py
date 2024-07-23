import streamlit as st

st.sidebar.page_link("streamlit_app.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/Job Postings Authoring.py", label="Job Postings Authoring", icon="💼")
st.sidebar.page_link("pages/Smart_ATS.py", label="Smart ATS", icon="📑")
st.sidebar.page_link("pages/Chatbot.py", label="Chatbot", icon="🤖")
st.sidebar.page_link("pages/Chatbot_FAQ.py", label="Help Desk", icon="📜")


welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of three of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  ##### Situation:
  A prominent HR firm headquartered in London, England helps employers find suitable candidates for job vacancies and assists individuals 
  in finding employment opportunities that fit their skills and career interests.They also specialize in helping clients reshape workforces 
  and deal with talent shortages.They have contacted Techment to implement a POC to demonstrate how employees can use Generative AI assistance 
  to help improve productivity and elevate employee experience while keeping their data safe.


  Use Case 1:  Job Postings Authoring
  Create engaging job posting descriptions that convey a position’s requirements, qualifications, and success criteria.
  
  
  Use Case 2: Keyword Search/Resumes 
  Be able to locate a list of resumes matching certain keywords, qualifications, geographies, job duties, etc. 
  that quickly helps narrow the search for candidates.  Be able to match a list of resumes to job postings.
    

  Use Case 3: Help Desk Knowledge Base: QnA
  Be able to construct knowledge base articles to answer frequently asked questions and or ask questions of a set of knowledgebase articles.
 
  
  ##### Solution:
  💼Job Postings Authoring: Created AI assistant by Using OCI Generative AI service.
  To Create engaging job posting descriptions we need to enter details about position’s requirements, qualifications, and success criteria.
  and then click on Generate Job posting button.


  📑Smart ATS(Application Tracking System): It will helps HR to Compare Job description with number of resumes
  to check Job Percentage Match with Job description, Matching keywords and Missing keywords,Profile Summary and Reason for percentage match. 
  
  
  🤖Chatbot: It will help Keyword Search/Resumes. We need upload resumes and ask questions.
  
  
  📜Help Desk Knowledge Base QnA: We need upload policy document pdf file ask questions.


  Please make sure to check the sidebar for more useful information. 💡
"""

st.write(welcome_message)