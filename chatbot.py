import streamlit as st
from transformers import pipeline
from io import StringIO
import fitz  # PyMuPDF for PDF handling
from docx import Document

# Load models with caching to improve performance
@st.cache_resource
def load_models():
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return question_generator, summarizer

question_generator, summarizer = load_models()

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":  # Text file
        return StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif uploaded_file.type == "application/pdf":  # PDF file
        text = ""
        pdf_document = fitz.open("pdf", uploaded_file.read())
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # Word file (.docx)
        doc = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    else:
        st.error("Unsupported file type.")
        return None

def generate_questions(text):
    input_text = f"generate questions: {text}"
    questions = question_generator(input_text, max_length=64, num_return_sequences=5)
    return [q['generated_text'] for q in questions]

def summarize_text(text):
    if len(text) > 1000:  # Truncate text if too long
        text = text[:1000]  # Limit to first 1000 characters
    summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to set a background image
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image with the provided link
set_background_image("https://wallpaperset.com/w/full/4/9/7/500747.jpg")

# App title with emoji and color styling
st.markdown("<h1 style='color: #ff69b4; text-align: center;'>🎀 CV Interview Prep Chatbot 🎀</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Upload your CV, and this chatbot will generate cute interview questions based on it. 💖💼</p>", unsafe_allow_html=True)

# File uploader with multiple file type support
st.markdown("<h3 style='color: #ff69b4;'>💌 Upload your CV (PDF, Word, or Text):</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["txt", "pdf", "docx"])

# Display stickers if file is uploaded
if uploaded_file is not None:
    # Extract text from the uploaded file
    cv_text = extract_text_from_file(uploaded_file)
    
    if cv_text:
        try:
            # Generate a summary of the CV
            summary = summarize_text(cv_text)
            st.markdown("<h3 style='color: #ff69b4;'>✨ Summary of your CV:</h3>", unsafe_allow_html=True)
            st.write(summary)

            # Generate interview questions based on the summary
            st.markdown("<h3 style='color: #ff69b4;'>📝 Generated Interview Questions:</h3>", unsafe_allow_html=True)
            questions = generate_questions(summary)
            for i, question in enumerate(questions):
                st.write(f"{i + 1}. {question} 💬")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    else:
        st.error("The uploaded file does not contain readable text.")
else:
    st.markdown("<p style='color: #ff69b4; text-align: center;'>Please upload your CV to start generating questions!</p>", unsafe_allow_html=True)

# Cute stickers section
st.markdown("<div style='text-align: center;'>🦄 🌈 🎉 🍓 🌸 🍩 🍪</div>", unsafe_allow_html=True)
