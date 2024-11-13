import streamlit as st
from langchain_groq import ChatGroq  # Import ChatGroq for interaction with LLM

# Initialize ChatGroq
model = ChatGroq(model_name="llama3-70b-8192")  # Replace with the correct model name if different

# Function to generate BRD using the model
def generate_brd(requirements, template_format):
    prompt = f"Generate a Business Requirements Document (BRD) in the following format:\n\n{template_format}\n\nBased on these requirements:\n{requirements}\n"
    response = model.generate(prompt)
    return response['generation']

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD"):
    if requirements and template_format:
        brd_content = generate_brd(requirements, template_format)
        st.subheader("Generated BRD:")
        st.write(brd_content)
    else:
        st.warning("Please enter both requirements and template format.")
