import streamlit as st
from langchain.prompts import PromptTemplate
import requests

# Groq API base URL
GROQ_API_URL = "https://api.groq.com/openai/v1/models/{Llama3-70b-8192}"  # Modify based on Groq API docs

# Initialize API key
API_KEY = "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"

# Define the PromptTemplate for the BRD
prompt_template = PromptTemplate(
    template="Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}",
    input_variables=['template_format', 'requirements']
)

# Define a function to generate the BRD using the Groq API
def generate_brd(requirements, template_format):
    # Format the prompt
    prompt = prompt_template.format(template_format=template_format, requirements=requirements)
    
    # Create the payload for the request
    data = {
        "inputs": [prompt],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Send POST request to Groq API
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['text']
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    with st.spinner("Generating..."):
        output = generate_brd(requirements=requirements, template_format=template_format)
        if output:
            st.write(output)
