import streamlit as st
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
import requests

# Custom Groq LLM class
class GroqLlama3LLM(LLM):
    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192", api_url: str = "https://api.groq.com/llm"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stop": stop,
        }

        response = requests.post(self.api_url, json=payload, headers=headers)
        response_data = response.json()

        if response.status_code == 200 and "generation" in response_data:
            return response_data["generation"]
        else:
            raise ValueError(f"Error from Groq API: {response_data}")

    @property
    def _llm_type(self) -> str:
        return "groq_llama3"

# Initialize GroqLlama3LLM with the API key
api_key = "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"
model = GroqLlama3LLM(api_key=api_key)

# Define the prompt template
prompt_template = PromptTemplate(
    template="Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}",
    input_variables=['template_format', 'requirements']
)

# Streamlit UI setup
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    # Format the prompt with user input
    formatted_prompt = prompt_template.format(template_format=template_format, requirements=requirements)
    
    # Call the model with the formatted prompt
    try:
        output = model._call(formatted_prompt)
        st.write("### Generated Business Requirements Document")
        st.write(output)
    except Exception as e:
        st.error(f"An error occurred: {e}")
