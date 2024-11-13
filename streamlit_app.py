import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Initialize the ChatGroq model
groq_model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

# Define the PromptTemplate for the BRD
prompt_template = PromptTemplate(
    template="Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}",
    input_variables=['template_format', 'requirements']
)

# Define a function to generate the BRD using ChatGroq model
def generate_brd(requirements, template_format):
    # Format the prompt
    prompt = prompt_template.format(template_format=template_format, requirements=requirements)
    
    # Assuming ChatGroq uses generate or a similar method to get the response
    result = groq_model.generate(prompt)  # Or groq_model.complete() depending on actual API
    return result['text']  # Assuming the result is a dictionary with a 'text' key

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
        st.write(output)
