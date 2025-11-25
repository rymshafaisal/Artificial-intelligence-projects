import streamlit as st # importing the Streamlit library for building web applications
import google.generativeai as genai # importing the Google Generative AI library for accessing Gemini models
import PyPDF2 as pdf # importing the PyPDF2 library for reading PDF files
import os # importing the os module for interacting with the operating system
from dotenv import load_dotenv # importing the load_dotenv function to load environment variables from a .env file
import json # importing the json module for working with JSON data
import re # importing the re module for regular expression operations

load_dotenv() # loading environment variables from the .env file

#configure GOOgle Generative AI with the API key environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # setting the API key for Google Generative AI

def get_gemini_response(prompt): # defining a function to get a response from the Gemini model

    model = genai.GenerativeModel("gemini-2.0-flash") # initializing the Gemini model with the specified version
    response = model.generate_content(prompt)
    return response.text # returning the generated text from the model response

def clean_json_response(response): # defining a function to clean and extract JSON from the model response
    # search for json-like structure in the response text using regex
    json_match = re.search(r'\{.*\}', response, re.DOTALL) # using regex to find the first occurrence of a JSON-like structure in the response text

    if json_match: # checking if a match is found
        #extract the matched JSON string
        json_str = json_match.group(0) # getting the matched JSON string from the response

        # clean up formatting issues by removing newlines and carriage returns
        json_str = json_str.replace('\n', ' ').replace('\r', '') # removing new
        return json_str # returning the cleaned JSON string
    
    return None

def input_pdf_text(file): # defining a function to extract text from a PDF file
    reader = pdf.PdfReader(file) # creating a PDF reader object to read the uploaded PDF file
    text = "" # initializing an empty string to store the extracted text
    for page in range(len(reader.pages)): # iterating through each page of the PDF file
        page = reader.pages[page] # getting the current page
        text += str(page.extract_text()) # extracting text from the current page and appending it to the text string
    
    return text # returning the extracted text from the PDF file


# Define the input prompt for the Gemini model
input_prompt = """ 
You are an expert ATS (Applicant Tracking System) with deep knowledge in tech fields including software engineering, data science, data analytics and machine learning.

Your task is to analyze the resume against the given job description and 
provide a deatiled analysis of how well the resume matches the job description.
consider the competitive job market and provide a actionable improvement suggestions .

IMPORTANT: You must respond with ONLY a JSON object in the following exact format, with no additional text:
{{
    "JD Match": "X%", 
    "MissingKeywords": ["keyword1", "keyword2", ...],
    "profileSummary": "your detailed analysis and improvement suggestions here",

}}

resume: {text}
description: {jd}

"""


st.set_page_config(page_title="Gemini ATS Resume Builder", page_icon=":guardsman:", layout="wide") # setting the page configuration for the Streamlit app

st.title("Gemini ATS Resume Builder") # setting the title of the Streamlit app
st.text("Improve your Resume ATS") # adding a description text to the app

# create a text area for job description input
jd = st.text_area("Paste the Job Description here") # creating a text area for the user to input the job description

# create a file uploader for pdf resume with help text
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf", help="Upload your resume in PDF format") # creating a file uploader for the user to upload their resume in PDF format


submit = st.button("Submit") # creating a submit button for the user to submit the job description and resume

if submit: # checking if the submit button is clicked
    if uploaded_file is not None and jd: # checking if a file is uploaded and job description is provided
        # extract text from the uploaded PDF file
        text = input_pdf_text(uploaded_file) # calling the function to extract text from the uploaded PDF file

        # format the prompt with resume text and job description
        prompt = input_prompt.format(text=text, jd=jd) # formatting the prompt with the extracted resume text and job description

        # call the Gemini model with the formatted prompt
        response = get_gemini_response(prompt) # generating text using the Gemini model with the formatted

        # clean and extract the JSOn response
        json_str = clean_json_response(response) # calling the function to clean and extract the JSON response

        # if json string is found in the response
        if json_str:
            try:
                # parse the JSON string into a python dictionary
                result = json.loads(json_str) # parsing the cleaned JSON string into a Python dictionary
                # display the analysis results header
                st.header("Analysis Results") # setting the header for the analysis results section
                # display the JD match percentage
                st.write(f"**JD Match:** {result['JD Match']}") # displaying the JD match percentage
                # display the missing keywords
                st.write("**Missing Keywords:**") # setting the header for the missing keywords section
                for keyword in result['MissingKeywords']: # iterating through the missing keywords
                    st.write(f"- {keyword}") # displaying each missing keyword
                # display the profile summary
                st.write("**Profile Summary:**") # setting the header for the profile summary section
                st.write(result['profileSummary']) # displaying the profile summary from the JSON response
            except json.JSONDecodeError: # handling JSON decoding errors
                st.error("Error parsing the JSON response. Please check the model output.") # displaying an error message if JSON parsing fails
        else: # if no JSON string is found in the response  
            st.error("No valid JSON response found. Please check the model output.")
    else: # if no file is uploaded or job description is not provided   
        st.error("Please upload a PDF resume and provide a job description.")

