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

# Initialize the LLMChain using a function to call the Groq model directly
def generate_brd(requirements, template_format):
    prompt = prompt_template.format(template_format=template_format, requirements=requirements)
    return groq_model.chat(prompt)

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
