import streamlit as st
from langchain.prompts import PromptTemplate
import groq

# Initialize the Groq client with your API key
groq_api_key = "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"  # Your Groq API key
groq_client = groq.Client(api_key=groq_api_key)

# Define the PromptTemplate for the BRD
prompt_template = PromptTemplate(
    template="Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}",
    input_variables=['template_format', 'requirements']
)

# Define a function to generate the BRD using the Groq model
def generate_brd(requirements, template_format):
    # Format the prompt using the PromptTemplate
    prompt = prompt_template.format(template_format=template_format, requirements=requirements)

    try:
        # Create the request body for Groq API
        request_data = {
            "model": "Llama3-70b-8192",  # Replace with the correct model name if necessary
            "inputs": [prompt],
        }
        
        # Generate response using the Groq client
        response = groq_client.generate(request_data)
        
        # Extract the generated text from the response
        result = response['data'][0]['generated_text']
        return result
    
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
