# Use streamlit to program GUI
# To run in terminal: uv run streamlit run main.py

import streamlit as st
import PyPDF2
import io
import os 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # load environment variables from .env file

st.set_page_config(page_title="AI CV Critiquer by Anh Duc Nguyen", # configure name of our tab/page
                   page_icon=":guardsman:",      #iIcon of the page
                   layout="centered"             # put content at center of screen
                   ) 

st.title("AI CV Critiquer :guardsman:") # Title of the app
st.markdown("Upload your CV in PDF format and get instant AI-powered feedback to improve it to your needs!") # Subtitle of the app

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # get API key from environment variable

# Here we're not building AI Agent but just a tool => need to load key like this for invoking LLM directly vs calling OpenAI agent

uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"]) # upload file button, file uploaded is stored in uploaded_file

job_role = st.text_input("Enter the job role you are applying for (optional):") # input box for job role, stored in job_role

analyze = st.button("Analyze CV") # Analyze button. Once pressed, analyze changed to True

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read())) # take uploaded file, read in and convert read info into Bytes object    
    return uploaded_file.read().decode("utf-8") # if txt file, read in and decode to utf-8

if analyze and uploaded_file: # ensure file is uploaded before analyzing
    st.write("Analyzing your CV...") # Show message while analyzing
    try:
        file_content = extract_text_from_file(uploaded_file) # extract text from uploaded file

        if not file_content.strip(): # remove whitespace and check if empty
            st.error("The uploaded file is empty. Please upload a valid PDF or TXT file.")
            st.stop()

        prompt = f"""Please analyze this CV and provide constructive feedback.
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else "general job applications"}

        Here is the CV content:
        {file_content}

        Provide your feedback in a clear and structured manner and specific recommendation.""" 

        client = OpenAI(api_key=OPENAI_API_KEY) # create OpenAI client
        response = client.chat.completions.create( # create chat completion
            model="gpt-3.5-turbo", # use gpt-3.5-turbo model
            messages=[
                {"role": "system", # harcoded system message for LLM
                 "content": "You are are an expert CV reviewer and career coach with years of experience in HR and recruitment."}, 
                {"role": "user", # user input based message containing the prompt, also prestructued but with inputs from user as above e.g. job_role
                 "content": prompt}
            ],
            max_tokens=500, # limit response to 500 tokens
            temperature=0.7 # set temperature for creativity
        )   

        st.markdown("### AI Feedback:") # Display feedback header
        st.markdown(response.choices[0].message.content) # Display AI feedback from response. Do this because we might have multiple choices in response
        # Use markdown for nice formatting

    except Exception as e:
        st.error(f"An error occurred: {e}")


# Room for improvements: add a second file uploader for job description and have AI compare CV to job description - TODO
        



