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
    # Split headers from the BRD format
    headers = [header.strip() for header in template_format.split("\n") if header.strip()]

    # Create a Word document
    doc = Document()
    doc.add_heading('Business Requirements Document', level=1)

    # Iterate through headers and match them with the requirements
    for header in headers:
        # Add header in bold
        paragraph = doc.add_paragraph()
        bold_run = paragraph.add_run(header)
        bold_run.bold = True

        # Check for exact matches in the requirements document
        exact_match_content = []
        for line in requirements.split("\n"):
            if header.lower() in line.lower():
                exact_match_content.append(line)

        # If there's an exact match, use it
        if exact_match_content:
            doc.add_paragraph("\n".join(exact_match_content), style='Normal')
        else:
            # Generate content for other headers using LLM
            prompt_input = {
                "template_format": header,
                "requirements": requirements,
            }
            generated_content = llm_chain.run(prompt_input)
            doc.add_paragraph(generated_content.strip(), style='Normal')

    # Save the Word document to a buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # Streamlit download button
    st.download_button(
        label="Download BRD as Word document",
        data=buffer,
        file_name="BRD.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

