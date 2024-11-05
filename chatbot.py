import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF for PDF handling
from docx import Document

# Load models with caching
@st.cache_resource(show_spinner=False)
def load_models():
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return question_generator, summarizer

# Function to summarize text
def summarize_text(text, summarizer):
    if len(text) < 20:  # Ensure text is long enough
        st.error("The text is too short for summarization. Please provide a longer document.")
        return ""
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_reader:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from Word documents
def extract_text_from_docx(uploaded_file):
    try:
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from Word document: {e}")
        return ""

# Function to generate questions from the summarized text
def generate_questions(text, question_generator):
    if not text.strip():
        st.error("No text available for question generation.")
        return []
    questions = []
    try:
        # Generate one question at a time
        for sentence in text.split('. '):  # Split into sentences for question generation
            if sentence:
                question = question_generator(sentence, max_length=50, do_sample=False)
                questions.append(question[0]['generated_text'])
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

# Set up Streamlit app
st.set_page_config(page_title="Interview Prep Chatbot", page_icon="💼")
st.title("Interview Prep Chatbot 🤖")
st.markdown("Upload your CV (PDF or Word) and get interview questions!")

# File uploader for PDF and Word files
uploaded_file = st.file_uploader("Choose a file...", type=["pdf", "docx"], label_visibility="visible")

if uploaded_file is not None:
    # Load models
    question_generator, summarizer = load_models()

    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        cv_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        cv_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        cv_text = ""

    if cv_text:
        # Summarize the CV text
        summary = summarize_text(cv_text, summarizer)
        st.subheader("CV Summary:")
        st.write(summary)

        # Generate interview questions
        questions = generate_questions(summary, question_generator)
        st.subheader("Generated Interview Questions:")
        for question in questions:
            st.write(f"- {question}")

st.markdown("Made with ❤️ by Your Name")
