import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Initialize ChatGroq
model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

prompt_template = PromptTemplate(
    template = f"Generate a Business Requirements Document (BRD) in the following format:\n\n{template_format}\n\nBased on these requirements:\n{requirements}\n",
    input_variables = ['template_format', 'requirements']
)

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    output = model(prompt_template.format(template_format = template_format, requirements = requirements))
    st.write(output)
