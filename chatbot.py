import streamlit as st
from transformers import pipeline
from io import StringIO

# Cache the model loading to improve performance
@st.cache_resource
def load_models():
    # Load question generation and summarization pipelines
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return question_generator, summarizer

question_generator, summarizer = load_models()

def generate_questions(text):
    input_text = f"generate questions: {text}"
    questions = question_generator(input_text, max_length=64, num_return_sequences=5)
    return [q['generated_text'] for q in questions]

# Streamlit app layout
st.title("CV Interview Prep Chatbot")
st.write("Upload your CV (in text format), and this chatbot will generate interview questions based on it.")

# File uploader
uploaded_file = st.file_uploader("Choose a CV file", type="txt")

if uploaded_file is not None:
    # Read uploaded file content
    cv_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    
    # Generate a summary of the CV
    summary = summarizer(cv_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    st.subheader("Summary of your CV:")
    st.write(summary)

    # Generate interview questions based on the summary
    questions = generate_questions(summary)
    st.subheader("Generated Interview Questions:")
    for i, question in enumerate(questions):
        st.write(f"{i + 1}. {question}")
