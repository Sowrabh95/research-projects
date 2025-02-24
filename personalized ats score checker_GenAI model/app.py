#import the python libraries
import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

## load all the environment variables
load_dotenv() 

# get your api key from this website by creating an account https://aistudio.google.com/apikey
#paste your api key in the .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#     Generate a response using the Gemini generative model.
#     Args: The input model name for which the response is to be generated.
#     Returns: The generated response text from the Gemini model.

def get_gemini_repsonse(input):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input)
    return response.text

#    Extract text from the uploaded pdf file.
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#Prompt Template: You are free to use any prompt template you wish.
input_prompt="""
You are an advanced AI specializing in ATS(Application Tracking System)
with a deep expertise in tech field, software engineering, data science, data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider that the job market is highly competitive and your evaluation should be rigorous, 
you should provide best assistance to improve the resume. Assign the percentage Matching based 
on job description (Jd) and the missing keywords with high accuracy
resume:{text}
description:{jd}

I want the response having the structure
{{"JD Match":"%", 
"Missing Keywords:[]", "Profile Summary":""
}}

"""

## streamlit app UI
st.title("Personalized ATS score checker")
st.text("Improve Your Resume ATS")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf document")

submit = st.button("Submit")

# if the submit button is clicked and the uploaded file is not None
# then extract the text from the pdf file and get the response from the gemini or the utilized model
if submit:
    if uploaded_file is not None:
        text=input_pdf_text(uploaded_file)
        response=get_gemini_repsonse(input_prompt)
        st.subheader(response)