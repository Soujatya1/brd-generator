import streamlit as st
from langchain_groq import ChatGroq  # Import ChatGroq for interaction with LLM

# Initialize ChatGroq
model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

# Function to generate BRD using the model
def generate_brd(requirements, template_format):
    # Construct the prompt
    prompt = f"Generate a Business Requirements Document (BRD) in the following format:\n\n{template_format}\n\nBased on these requirements:\n{requirements}\n"
    
    # Pass the prompt in a list as messages
    messages = [prompt] # Just a list containing the prompt text
    
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
