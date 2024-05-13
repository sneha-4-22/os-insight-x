import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("CodeGPTPlus/deepseek-coder-1.3b-typescript")
model = AutoModelForCausalLM.from_pretrained("CodeGPTPlus/deepseek-coder-1.3b-typescript")


def generate_code_suggestions(input_code):
    inputs = tokenizer(input_code, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


def main():
    st.title("Code Quality Checker and Suggestions")
    code_input = st.text_area("Enter your code here:", height=200)
    if st.button("Check Code Quality and Get Suggestions"):
        with st.spinner("Generating Suggestions..."):
            suggestions = generate_code_suggestions(code_input)
            st.success("Suggestions Generated!")
            st.code(suggestions, language='typescript')

if __name__ == "__main__":
    main()
