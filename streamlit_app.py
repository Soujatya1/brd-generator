import streamlit as st
import requests

# Set up your API details
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/v1/models/llama3-70b-8192"  # Update with the correct endpoint if different

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    # Set up the prompt text
    prompt = f"Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}"
    
    # Make the API call to Groq
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 800  # Adjust as needed for the response length
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Check for request errors
        output = response.json().get("generation", {}).get("text", "Error: No response text found.")
        st.write(output)
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Groq API: {e}")
