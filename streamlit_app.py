import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

# Initialize ChatGroq
model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

# Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Initialize FAISS vector store
faiss_index = faiss.IndexFlatL2(768)  # The dimension of the embeddings

# Function to generate BRD using the model
def generate_brd(requirements, template_format):
    # Construct the prompt
    prompt = f"Generate a Business Requirements Document (BRD) in the following format:\n\n{template_format}\n\nBased on these requirements:\n{requirements}\n"
    
    # Embed the requirements and template format using HuggingFaceEmbeddings
    embedding_requirements = embedding_model.embed([requirements])
    embedding_template_format = embedding_model.embed([template_format])

    # Store the embeddings in FAISS (Note: FAISS needs numpy arrays)
    faiss_index.add(np.array(embedding_requirements).astype(np.float32))
    faiss_index.add(np.array(embedding_template_format).astype(np.float32))

    # Retrieve the closest vector (embedding) in FAISS (here, we'll just use the latest added)
    # You may want to enhance this retrieval logic based on your application
    _, closest_idx = faiss_index.search(np.array(embedding_requirements).astype(np.float32), k=1)

    # Retrieve relevant context from stored embeddings (for simplicity, returning the same input)
    # This could be expanded to retrieve the most relevant stored document from the FAISS index
    relevant_context = requirements if closest_idx[0][0] == 0 else template_format
    
    # Pass the prompt along with the relevant context in a list as messages
    messages = [prompt + "\n\nContext: " + relevant_context]  # Adding context from FAISS search
    
    # Call model.generate with messages
    response = model.generate(messages=messages, temperature=0.7, max_tokens=1000)
    
    # Extract the response text if 'generation' or a similar field is available
    return response.get('generation', 'No response generated')

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
