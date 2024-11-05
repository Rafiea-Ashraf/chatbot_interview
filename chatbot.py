from transformers import pipeline
from io import StringIO

# Load the models
@st.cache_resource
def load_models():
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return question_generator, summarizer

question_generator, summarizer = load_models()

# Helper function to generate interview questions
def generate_questions(text):
    input_text = f"generate questions: {text}"
    questions = question_generator(input_text, max_length=64, num_return_sequences=5)
    return [q['generated_text'] for q in questions]

# Streamlit app interface
st.title("CV Interview Prep Chatbot")

st.write("Upload your CV (in text format), and this chatbot will generate interview questions based on it.")

# Step 3: Upload CV
uploaded_file = st.file_uploader("Choose a CV file", type="txt")
if uploaded_file is not None:
    # Read CV file
    cv_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

    # Summarize CV content
    summary = summarizer(cv_text, max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    st.subheader("Summary of your CV:")
    st.write(summary)

    # Generate interview questions based on the summary
    questions = generate_questions(summary)
    st.subheader("Generated Interview Questions:")
    for i, question in enumerate(questions):
        st.write(f"{i + 1}. {question}")

    # Step 4: Chatbot interaction
    st.write("## Interview Chatbot")
    st.write("Type 'ask me a question' to get a new interview question.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You: ", "")
    if user_input:
        if user_input.lower() == "ask me a question":
            if questions:
                bot_response = questions.pop(0)
            else:
                bot_response = "No more questions left!"
        else:
            bot_response = "Chatbot: Sorry, I can only ask interview questions at the moment."

        # Display conversation history
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
        for chat in st.session_state.chat_history:
            st.write(f"You: {chat['user']}")
            st.write(f"Chatbot: {chat['bot']}")

