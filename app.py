import streamlit as st
from utils import extract_pdf_text, get_random_chunk
from mistral_api import generate_mcq_question, generate_essay_question, evaluate_essay_answer
import tempfile

st.set_page_config(page_title="AI-Powered Quiz Master", layout="centered")

st.title("📚 AI-Powered Quiz Master")

uploaded_file = st.file_uploader("Upload your study material (PDF only)", type="pdf")
question_type = st.radio("Choose your question type:", ["Multiple Choice", "Essay"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_text = extract_pdf_text(tmp.name)
        chunk = get_random_chunk(pdf_text)

    if st.button("Generate Question"):
        if question_type == "Multiple Choice":
            question, options, correct_option = generate_mcq_question(chunk)
            st.subheader(question)
            selected = st.radio("Options", options, key="mcq")
            if st.button("Submit Answer"):
                if selected == correct_option:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. Correct answer: {correct_option}")
        else:
            question = generate_essay_question(chunk)
            st.markdown("**Essay Question:**")
            st.write(question)
            student_answer = st.text_area("Your answer")
            if st.button("Evaluate Essay"):
                feedback = evaluate_essay_answer(chunk, question, student_answer)
                st.markdown("### Feedback:")
                st.write(feedback)
