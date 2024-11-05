import streamlit as st
from transformers import pipeline
import gc

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_question_generator():
    return pipeline("question-generation", model="valhalla/t5-small-qa-qg-hl")

def summarize_text(text, summarizer):
    if len(text) < 20:
        st.error("The text is too short for summarization. Please provide a longer document.")
        return ""
    
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        if summary:
            return summary[0]['summary_text']
        else:
            st.error("No summary generated.")
            return ""
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

def generate_questions(text, question_generator):
    try:
        questions = question_generator(text)
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

# Streamlit UI
st.title("Document Summarization and Question Generation App")
input_text = st.text_area("Paste your document text here:", height=300)

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        summarizer = load_summarizer()
        summary = summarize_text(input_text, summarizer)
        if summary:
            st.subheader("Summary:")
            st.write(summary)
        gc.collect()

if st.button("Generate Questions"):
    with st.spinner("Generating questions..."):
        question_generator = load_question_generator()
        questions = generate_questions(input_text, question_generator)
        if questions:
            st.subheader("Generated Questions:")
            for q in questions:
                st.write(f"- {q['question']}")

st.markdown("### Note: The models may take some time to load on first use. Please be patient.")
