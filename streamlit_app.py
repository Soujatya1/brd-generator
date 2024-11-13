import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from docx import Document
from io import BytesIO
import re  # For identifying topics and sub-topics

# Initialize ChatGroq
model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

# Modify the prompt template to include elaboration instruction
llm_chain = LLMChain(llm=model, prompt=PromptTemplate(
    input_variables=['template_format', 'requirements'],
    template = "Generate a detailed Business Requirements Document (BRD) in the following format: {template_format}. "
             "For each topics and sub-topics, provide thorough explanations and elaborate on each topic based on these requirements: {requirements}, please do not hallucinate"
))

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a detailed Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    # Generate the prompt
    prompt_input = {"template_format": template_format, "requirements": requirements}
    output = llm_chain.run(prompt_input)
    st.write(output)
    
    # Create a Word document
    doc = Document()
    doc.add_heading('Business Requirements Document', level=1)

    # Function to add bold formatting for topics and sub-topics
    def add_bold_paragraph(doc, text):
        # Example regex to match headings (can be customized based on the output format)
        # Assuming topics start with "1." or any numbered format or uppercase headings
        paragraphs = re.split(r'(\d+\.\s*[A-Z].*?)(?=\s*\d+\.|\Z)', text, flags=re.DOTALL)  # Split by numbered headings
        for para in paragraphs:
            if re.match(r'^\d+\.\s*[A-Z].*', para):  # If it's a topic or sub-topic
                # Add bolded topic/sub-topic
                doc.add_paragraph(para, style='Heading 2')
            else:
                # Add normal content
                doc.add_paragraph(para)
    
    # Add the BRD content with bold formatting for topics/sub-topics
    add_bold_paragraph(doc, output)

    # Save the document to a BytesIO buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # Provide download button
    st.download_button(
        label="Download BRD as Word document",
        data=buffer,
        file_name="BRD.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
