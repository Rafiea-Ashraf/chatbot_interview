import streamlit as st
from transformers import pipeline

# Load models (this may take some time on the first run)
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    question_generator = pipeline("question-generation", model="valhalla/t5-small-qa-qg-hl")
    return summarizer, question_generator

# Main Streamlit application
def main():
    st.title("Document Summarization and Question Generation App")

    input_text = st.text_area("Paste your document text here:", height=300)

    if st.button("Summarize"):
        if input_text.strip():
            summarizer, _ = load_models()
            with st.spinner("Summarizing..."):
                summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])
        else:
            st.error("Please enter some text for summarization.")

    if st.button("Generate Questions"):
        if input_text.strip():
            _, question_generator = load_models()
            with st.spinner("Generating questions..."):
                questions = question_generator(input_text)
            st.subheader("Generated Questions:")
            for q in questions:
                st.write(f"- {q['question']}")
        else:
            st.error("Please enter some text to generate questions.")

if __name__ == "__main__":
    main()
