import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from io import BytesIO
import docx  # For processing .docx files
import PyPDF2  # For processing .pdf files
import os

# Initialize the model
model = ChatGroq(
    groq_api_key="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", 
    model_name="Llama3-70b-8192"
)

llm_chain = LLMChain(llm=model, prompt=PromptTemplate(
    input_variables=['template_format', 'requirements'],
    template="Generate a detailed Business Requirements Document (BRD) in the following format: {template_format}. "
             "For each topics and sub-topics, provide thorough explanations and elaborate on each topic based on these requirements: {requirements}, please do not hallucinate"
))

st.title("BRD Generator")
st.write("Upload requirement documents and define the BRD structure below to generate a detailed Business Requirements Document.")

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload requirement documents (PDF/DOCX):", accept_multiple_files=True)

# Text area for BRD format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Function to extract text from .docx files
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Function to extract text from .pdf files
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Process uploaded files
if uploaded_files:
    combined_requirements = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
        if file_extension == ".docx":
            combined_requirements += extract_text_from_docx(uploaded_file) + "\n"
        elif file_extension == ".pdf":
            combined_requirements += extract_text_from_pdf(uploaded_file) + "\n"
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
    
    requirements = combined_requirements
else:
    requirements = ""

# Generate BRD when button is clicked
if st.button("Generate BRD") and requirements and template_format:
    # Generate the prompt
    prompt_input = {"template_format": template_format, "requirements": requirements}
    output = llm_chain.run(prompt_input)
    st.write(output)
    
    # Create a Word document
    doc = Document()
    doc.add_heading('Business Requirements Document', level=1)
    
    # Add the template format in bold
    doc.add_heading('Template Format:', level=2)
    paragraph = doc.add_paragraph()
    bold_run = paragraph.add_run(template_format)
    bold_run.bold = True  # Make the template format text bold

    # Add the generated BRD content
    doc.add_heading('Generated BRD:', level=2)
    doc.add_paragraph(output)

    # Save the Word document to a buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # Download button for Word document
    st.download_button(
        label="Download BRD as Word document",
        data=buffer,
        file_name="BRD.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
