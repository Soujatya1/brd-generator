import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from io import BytesIO
import hashlib
import PyPDF2
import os

# Initialize the model
model = ChatGroq(
    groq_api_key="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", 
    model_name="Llama3-70b-8192"
)

llm_chain = LLMChain(llm=model, prompt=PromptTemplate(
    input_variables=['template_format', 'requirements', 'tables'],
    template="Generate a detailed BRD in the following format: {template_format}. "
             "Whatever information is available in the {requirements} as per {template_format}, display those under each heading"
             "Include the following tabular content wherever applicable: {tables}."
))

st.title("BRD Generator")
st.write("Upload requirement documents and define the BRD structure below to generate a detailed Business Requirements Document.")

uploaded_files = st.file_uploader("Upload requirement documents (PDF/DOCX):", accept_multiple_files=True)
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Function to extract text and tables from .docx files
def extract_content_from_docx(file):
    doc = Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            table_data.append([cell.text.strip() for cell in row.cells])
        tables.append(table_data)
    # Convert tables to a string format
    tables_as_text = ""
    for table_data in tables:
        tables_as_text += "\n".join(["\t".join(row) for row in table_data]) + "\n\n"
    return text, tables_as_text

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
    all_tables_as_text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
        if file_extension == ".docx":
            text, tables_as_text = extract_content_from_docx(uploaded_file)
            combined_requirements += text + "\n"
            all_tables_as_text += tables_as_text + "\n"
        elif file_extension == ".pdf":
            combined_requirements += extract_text_from_pdf(uploaded_file) + "\n"
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
    
    requirements = combined_requirements
else:
    requirements = ""
    all_tables_as_text = ""

def generate_hash(template_format, requirements):
    combined_string = template_format + requirements
    return hashlib.md5(combined_string.encode()).hexdigest()

if "outputs_cache" not in st.session_state:
    st.session_state.outputs_cache = {}

# Generate BRD when button is clicked
if st.button("Generate BRD") and requirements and template_format:
    doc_hash = generate_hash(template_format, requirements)
    
    if doc_hash in st.session_state.outputs_cache:
        #st.write("Using cached BRD...")
        output = st.session_state.outputs_cache[doc_hash]
    else:
        prompt_input = {
            "template_format": template_format,
            "requirements": requirements,
            "tables": all_tables_as_text,
        }
        output = llm_chain.run(prompt_input)
        st.session_state.outputs_cache[doc_hash] = output
    st.write(output)

    # Create a Word document
    doc = Document()
    doc.add_heading('Business Requirements Document', level=1)
    doc.add_paragraph(output, style='Normal')

    # Append tabular content, if any
    

    # Save the Word document to a buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    st.download_button(
        label="Download BRD as Word document",
        data=buffer,
        file_name="BRD.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
