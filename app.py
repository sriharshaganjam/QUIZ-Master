import streamlit as st
import PyPDF2  # or import pdfplumber
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import json
import re

st.set_page_config(page_title="Quiz Generator", layout="centered")

st.title("📚 AI Quiz Master")
st.markdown("Generate questions from any content using Mistral + Sentence Transformers.")

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        return None

embedding_model = load_embedding_model()

mistral_api_key = st.secrets.get("mistral", {}).get("api_key")
mistral_api_base_url = st.secrets.get("mistral", {}).get("base_url", "https://api.mistral.ai/v1")

if not mistral_api_key:
    st.error("❌ Mistral API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")

@st.cache_resource(show_spinner=False)
def create_openai_client(api_key, base_url):
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

client = create_openai_client(mistral_api_key, mistral_api_base_url)

def clean_and_parse_json(raw_text):
    try:
        match = re.search(r"""```json\s*(\{.*?\})\s*```""", raw_text, re.DOTALL)
        if match:
            json_text = match.group(1)
            return json.loads(json_text)
        else:
            return json.loads(raw_text)
    except Exception as e:
        st.warning(f"⚠️ JSON parse error: {e}")
        return None

def get_cosine_similarity(source, response, model):
    if model is None:
        return 0.0
    emb1 = model.encode(source, convert_to_tensor=True)
    emb2 = model.encode(response, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
    return round(cosine_sim * 100, 2)

# Initialize session state variables if they don't exist
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'current_choices' not in st.session_state:
    st.session_state.current_choices = None
if 'user_answer_mc' not in st.session_state:
    st.session_state.user_answer_mc = None
if 'user_answer_essay' not in st.session_state:
    st.session_state.user_answer_essay = ""
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'question_similarity' not in st.session_state:
    st.session_state.question_similarity = 0.0

question_mode = st.selectbox("Question Type", ["Multiple Choice", "Essay"])

# Replace or augment the text_area with a file uploader
uploaded_file = st.file_uploader("Upload Educational Content (PDF or Text)", type=["txt", "pdf"])

input_text = ""  # Initialize input_text
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        try:
            # Use PyPDF2 or pdfplumber to extract text
            pdf_reader = PyPDF2.PdfReader(uploaded_file) # or pdfplumber.open(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                input_text += pdf_reader.pages[page_num].extract_text()
            st.success("PDF uploaded and text extracted!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            input_text = st.text_area("📘 Paste Educational Content (Text) if PDF failed or you prefer", height=200)
    elif uploaded_file.type == "text/plain":
        input_text = uploaded_file.read().decode("utf-8")
        st.success("Text file uploaded!")
else:
    input_text = st.text_area("📘 Paste Educational Content (Text)", height=200)

if st.button("🧠 Generate Question"):
    st.session_state.show_feedback = False  # Reset feedback when new question generated
    st.session_state.user_answer_mc = None
    st.session_state.user_answer_essay = ""

    if not input_text:
        st.warning("Please enter some text or upload a file.")
    elif not client:
        st.error("Mistral client not initialized.")
    else:
        with st.spinner("Generating question..."):
            prompt = f"""
        You are a quiz master. Based on the following text, generate a {"multiple choice" if question_mode == "Multiple Choice" else "short answer"} question in JSON format.
        Your response MUST be ONLY valid JSON, enclosed within a ```json block. Do not include any other text or explanations outside this block.

        Text:
        \"\"\"{input_text}\"\"\"

        Format for Multiple Choice:
        ```json
        {{
          "question": "What is the ...?",
          "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
          "answer": "B"
        }}
        ```

        Format for Essay:
        ```json
        {{
          "question": "Explain the ... in your own words.",
          "answer": "Expected points or summary"
        }}
        ```
        """

            try:
                response = client.chat.completions.create(
                    model="mistral-tiny",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7  # Increased temperature for more variation
                )

                mistral_raw = response.choices[0].message.content
                question_json = clean_and_parse_json(mistral_raw)

                if not question_json:
                    st.error("❌ Mistral did not return valid JSON or the JSON was empty.")
                    st.session_state.current_question = None
                else:
                    st.session_state.current_question = question_json.get('question')
                    st.session_state.current_answer = question_json.get('answer')
                    st.session_state.current_choices = question_json.get('choices')

                    st.session_state.question_similarity = get_cosine_similarity(input_text, st.session_state.current_question, embedding_model)

                    st.success("✅ Question Generated! Now try to answer.")
                    # Don't display answer here yet!
            except Exception as e:
                st.error(f"❌ Error during generation: {e}")
                st.session_state.current_question = None  # Clear question on error

# --- Display Question and Get User Answer ---
if st.session_state.current_question:
    st.markdown("---")
    st.markdown(f"**Question:** {st.session_state.current_question}")

    if st.session_state.current_choices:  # Multiple Choice
        st.session_state.user_answer_mc = st.radio(
            "Your Answer:",
            st.session_state.current_choices,
            index=None,  # No default selection
            key="mc_answer"  # Unique key for widget
        )
    else:  # Essay
        st.session_state.user_answer_essay = st.text_area(
            "Your Answer (100 words max):",  # Added word limit guidance
            value=st.session_state.user_answer_essay,  # Retain text if re-rendered
            max_chars=700, # Approximate 100 words (7 chars/word)
            key="essay_answer"
        )

    if st.button("✅ Evaluate"): # Changed button text from "Submit" to "Evaluate"
        st.session_state.show_feedback = True

# --- Display Feedback ---
if st.session_state.show_feedback:
    st.markdown("---")
    if question_mode == "Multiple Choice":
        if st.session_state.user_answer_mc:
            # Extract the letter from the choice (e.g., "A. Option" -> "A")
            selected_letter = st.session_state.user_answer_mc.split('.')[0]
            if selected_letter == st.session_state.current_answer:
                st.success("🎉 Correct Answer!")
            else:
                st.error(f"❌ Incorrect. The correct answer was: {st.session_state.current_answer}")
            st.info(f"**Correct Answer:** {st.session_state.current_answer}")  # Show the answer for review
        else:
            st.warning("Please select an answer.")
    else:  # Essay
        if st.session_state.user_answer_essay.strip():
            st.markdown(f"**Your Answer:** {st.session_state.user_answer_essay}")
            st.markdown(f"**Expected Answer:** {st.session_state.current_answer}")

            # Optional: Calculate similarity between user's essay answer and expected answer
            if embedding_model:
                user_answer_similarity = get_cosine_similarity(st.session_state.current_answer, st.session_state.user_answer_essay, embedding_model)
                st.info(f"**Your Answer Similarity to Expected:** {user_answer_similarity}%")

                # Provide feedback based on similarity
                if user_answer_similarity > 80:
                    st.success("Excellent! Your answer covers the key points.")
                elif user_answer_similarity > 60:
                    st.warning("Good, but some minor points might be missing. Review the expected answer.")
                else:
                    st.error("Your answer is missing significant points. Review the expected answer carefully.")

            else:
                st.warning("Could not calculate user answer similarity (embedding model not loaded).")
        else:
            st.warning("Please provide an answer.")

    # Always show question-text similarity after feedback
    st.markdown(f"🧠 **Question-Text Similarity:** {st.session_state.question_similarity}%")
