import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from io import BytesIO

# Initialize ChatGroq
model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

# Create an LLM chain
llm_chain = LLMChain(llm=model, prompt=PromptTemplate(
    template="Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}",
    input_variables=['template_format', 'requirements']
))

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    # Generate the prompt
    prompt_input = {"template_format": template_format, "requirements": requirements}
    output = llm_chain.run(prompt_input)
    
    # Display the generated BRD
    st.write(output)

    # Create a Word document with the output
    doc = Document()
    doc.add_heading("Business Requirements Document", level=1)
    doc.add_paragraph(output)
    
    # Save the document to a BytesIO object
    doc_buffer = BytesIO()
    doc.save(doc_buffer)
    doc_buffer.seek(0)
    
    # Provide download link for the Word document
    st.download_button(
        label="Download BRD as Word Document",
        data=doc_buffer,
        file_name="Business_Requirements_Document.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
