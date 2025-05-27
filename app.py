import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import random
import requests
import os

# Load Mistral API key from Streamlit secrets or environment variable
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY", "your_mistral_api_key_here")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Quiz Master", layout="centered")
st.title("📚 AI-Powered Quiz Master")

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload your study material (PDF only)", type=["pdf"])

if uploaded_file:
    reader = PdfReader(uploaded_file)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # Chunking text into ~500-character chunks
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
    embeddings = embedder.encode(chunks, convert_to_tensor=True)

    question_type = st.radio("Choose your question type:", ["Multiple Choice", "Essay"])

    if st.button("Generate Question"):
        random_chunk = random.choice(chunks)

        if question_type == "Multiple Choice":
            # Simulated MCQ question
            question = f"What is a key point from the following excerpt?\n\n{random_chunk}"
            options = [
                "Correct: " + random_chunk[:40] + "...",
                "Distractor A",
                "Distractor B",
                "Distractor C"
            ]
            random.shuffle(options)
            selected = st.radio(question, options)

            if st.button("Submit Answer"):
                if "Correct" in selected:
                    st.success("✅ Correct answer!")
                else:
                    st.error("❌ Incorrect. Try again!")

        elif question_type == "Essay":
            st.markdown(f"**Essay Question:**\n\nBased on the following text, write a short answer.\n\n{random_chunk}")
            user_answer = st.text_area("Your answer")

            if st.button("Evaluate Essay") and user_answer:
                with st.spinner("Evaluating with Mistral..."):
                    prompt = f"""
You are an intelligent tutor. Here is a passage from study material:

\"\"\"{random_chunk}\"\"\"

A student answered the question based on it:

\"\"\"{user_answer}\"\"\"

Evaluate the student’s answer for accuracy, completeness, and relevance to the above content. 
Provide a score out of 10 and a short explanation.
"""
                    try:
                        response = requests.post(
                            MISTRAL_URL,
                            headers={
                                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "mistral-tiny",
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": 0.5
                            }
                        )
                        if response.status_code == 200:
                            reply = response.json()["choices"][0]["message"]["content"]
                            st.success("✅ Evaluation complete:")
                            st.markdown(reply)
                        else:
                            st.error("❌ Mistral API call failed.")
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")
else:
    st.info("Please upload a PDF file to begin.")
