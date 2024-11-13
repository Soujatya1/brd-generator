import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Initialize ChatGroq
model = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")

prompt_template = PromptTemplate(
    template="Generate a Business Requirements Document (BRD) in the following format: {template_format} based on these requirements: {requirements}",
    input_variables=['template_format', 'requirements']
)

# Streamlit UI
st.title("BRD Generator")
st.write("Enter the details below to generate a Business Requirements Document.")

# Input section for requirements
requirements = st.text_area("Enter the requirements:", height=200, placeholder="List the business requirements here...")

# Input section for BRD template format
template_format = st.text_area("Enter the BRD format:", height=200, placeholder="Define the structure of the BRD here...")

# Generate BRD button
if st.button("Generate BRD") and requirements and template_format:
    try:
        # Generate the formatted prompt
        formatted_prompt = prompt_template.format(template_format=template_format, requirements=requirements)

        # Debugging: Check the type and content of the formatted prompt
        st.write("Formatted Prompt Type:", type(formatted_prompt))  # Check the type
        st.write("Formatted Prompt Content:", formatted_prompt)  # Check the content

        # Call the model with the formatted prompt
        output = model(formatted_prompt)
        
        # Debugging: Check the type and content of the output
        st.write("Model Output Type:", type(output))  # Check the output type
        st.write("Model Output Content:", output)  # Check the output content

        # Handle the output type correctly
        if isinstance(output, dict):
            # If output is a dictionary, check if it has a 'generation' field
            generated_text = output.get('generation', None)
            if generated_text:
                st.write(generated_text)  # Display the BRD
            else:
                st.write("No 'generation' field found in the output.")
        else:
            # If output is a string, just display it
            st.write(output)

    except Exception as e:
        st.error(f"Error generating BRD: {e}")
else:
    if not requirements or not template_format:
        st.warning("Please enter both requirements and template format before generating the BRD.")
