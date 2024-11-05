import streamlit as st
from transformers import pipeline
from io import BytesIO
import fitz  # PyMuPDF for PDF handling
from docx import Document
import numpy as np

# Load models only when needed to optimize memory usage
@st.cache_resource
def load_models():
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return question_generator, summarizer

# Function to summarize text
def summarize_text(text):
    # Limit the text to the first 500 characters for summarization
    text = text[:500]
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    return text

# Function to extract text from Word documents
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to generate questions from the summarized text
def generate_questions(text):
    questions = question_generator(text, max_length=50, num_return_sequences=5, do_sample=False)
    return [q['generated_text'] for q in questions]

# Set up Streamlit app
st.set_page_config(page_title="Interview Prep Chatbot", page_icon="💼")
st.title("Interview Prep Chatbot 🤖")
st.markdown("Upload your CV (PDF or Word) and get interview questions!")

# Image for the app
st.image("https://wallpaperset.com/w/full/4/9/7/500747.jpg", use_column_width=True)

# File uploader for PDF and Word files
uploaded_file = st.file_uploader("Choose a CV file (PDF or Word)", type=["pdf", "docx"])

if uploaded_file:
    # Load models
    question_generator, summarizer = load_models()
    
    # Process the uploaded file
    if uploaded_file.type == "application/pdf":
        cv_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        cv_text = extract_text_from_docx(uploaded_file)

    # Summarize the CV text
    if cv_text:
        summary = summarize_text(cv_text)
        st.subheader("Summary of your CV:")
        st.write(summary)

        # Generate questions based on the summary
        questions = generate_questions(summary)
        st.subheader("Potential Interview Questions:")
        for question in questions:
            st.write(f"- {question}")

# Add cute stickers and colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff; /* Light blue background */
    }
    h1 {
        color: #ff4500; /* Orange */
    }
    h2 {
        color: #ff69b4; /* Pink */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🎉 Welcome to the Interview Prep Chatbot!")
st.sidebar.markdown("Upload your CV and get prepared for your next interview!")

# Add a cute sticker image in the sidebar
st.sidebar.image("https://img.icons8.com/ios/50/000000/clipboard.png", width=50)
