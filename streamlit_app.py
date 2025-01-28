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
    template="Create a Business Requirements Document (BRD) based on the following details:
    
Document Structure:
{template_format}

Requirements:
Analyze the content provided in the requirement documents and map the relevant information to each section defined in the BRD structure. Be concise and specific.

Tables:
If applicable, include the following tabular information extracted from the documents:
{tables}

Formatting:
1. Use headings and subheadings for clear organization.
2. Include bullet points or numbered lists where necessary for better readability.
3. Clearly differentiate between functional and non-functional requirements.
4. Provide tables in a well-structured format, ensuring alignment and readability.

Key Points:
1. Use the given format `{template_format}` strictly as the base structure for the BRD.
2. Ensure all relevant information from the requirements is displayed under the corresponding section.
3. Avoid including irrelevant or speculative information.
4. Summarize lengthy content while preserving its meaning.

Output:
The output must be formatted cleanly as a Business Requirements Document, following professional standards. Avoid verbose language and stick to the structure defined above.
"
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
output = st.session_state.outputs_cache

if isinstance(output, dict) and 'text' in output:
    output_text = output['text']
else:
    output_text = str(output)
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Function to calculate text similarity using Cosine Similarity
def calculate_text_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1] * 100  # Return as a percentage

# Function to calculate structural similarity
def calculate_structural_similarity(tables1, tables2):
    sm = difflib.SequenceMatcher(None, tables1, tables2)
    return sm.ratio() * 100  # Return as a percentage

# New section: Upload a sample document for comparison
st.write("Optional: Upload a sample BRD for comparison.")
sample_file = st.file_uploader("Upload a sample BRD (PDF/DOCX):", type=["pdf", "docx"])

if sample_file:
    file_extension = os.path.splitext(sample_file.name)[-1].lower()
    if file_extension == ".docx":
        sample_text, sample_tables = extract_content_from_docx(sample_file)
    elif file_extension == ".pdf":
        sample_text = extract_text_from_pdf(sample_file)
        sample_tables = ""  # No table support in the current PDF extraction function
    else:
        st.warning(f"Unsupported file format: {sample_file.name}")
        sample_text, sample_tables = "", ""

    # Ensure there's content to compare
    if requirements and template_format and sample_text:
        # Calculate match scores
        content_similarity = calculate_text_similarity(output_text, sample_text)
        format_similarity = calculate_structural_similarity(all_tables_as_text, sample_tables)
        
        # Final weighted score
        content_weight = 0.7
        format_weight = 0.3
        final_score = (content_similarity * content_weight) + (format_similarity * format_weight)

        # Display results
        st.subheader("Match Score Results")
        st.write(f"Content Match: {content_similarity:.2f}%")
        st.write(f"Format Match: {format_similarity:.2f}%")
        st.write(f"Overall Match Score: {final_score:.2f}%")
    else:
        st.warning("Please generate a BRD first or ensure the sample document has valid content.")
