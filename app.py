# app.py

import streamlit as st
import fitz  # PyMuPDF
from mistral_api import call_mistral

st.title("📚 AI Quiz Master")
st.caption("Upload a PDF, choose a question type, and let the AI quiz you!")

uploaded_file = st.file_uploader("Upload your study material (PDF only)", type="pdf")

question_type = st.radio("Choose your question type:", ["Multiple Choice", "Essay"])
generate_btn = st.button("Generate Question")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def generate_essay_question(chunk):
    prompt = f"""
You're an exam designer. Based on the passage below, create a challenging essay-style question that tests critical thinking or understanding.

Text:
\"\"\"{chunk}\"\"\"

Respond with only the question.
"""
    return call_mistral(prompt)

def generate_mcq_question(chunk):
    prompt = f"""
You're a teacher creating a multiple choice question based on the following text. Write:

1. A clear and concise question that tests key comprehension.
2. Four answer options labeled A, B, C, D.
3. Identify which one is correct.

Text:
\"\"\"{chunk}\"\"\"

Format:
Question: <question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Correct Answer: <A/B/C/D>
"""
    response = call_mistral(prompt)
    lines = response.strip().split('\n')

    try:
        question = lines[0].replace("Question: ", "")
        options = {
            "A": lines[1][3:].strip(),
            "B": lines[2][3:].strip(),
            "C": lines[3][3:].strip(),
            "D": lines[4][3:].strip(),
        }
        correct_option = lines[5].split(":")[-1].strip()
        return question, options, correct_option
    except Exception as e:
        st.error("⚠️ Failed to parse the model's response. Please try with a different PDF.")
        return "", {}, ""

if generate_btn and uploaded_file:
    with st.spinner("Generating question..."):
        content = extract_text_from_pdf(uploaded_file)
        chunk = content[:1000]  # Simplified chunking

        try:
            if question_type == "Essay":
                question = generate_essay_question(chunk)
                st.subheader("📝 Essay Question")
                st.write(question)

            else:
                question, options, correct = generate_mcq_question(chunk)
                st.subheader("❓ Multiple Choice Question")
                st.write(question)
                for k, v in options.items():
                    st.write(f"{k}. {v}")
                st.success(f"✅ Correct Answer: {correct}")
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")
