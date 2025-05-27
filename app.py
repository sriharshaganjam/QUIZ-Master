import streamlit as st
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import json

# --- Load SentenceTransformer embedding model ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        return None

embedding_model = load_embedding_model()

# --- Mistral API credentials from Streamlit secrets ---
mistral_api_key = st.secrets.get("mistral", {}).get("api_key")
mistral_api_base_url = st.secrets.get("mistral", {}).get("base_url", "https://api.mistral.ai/v1")

if not mistral_api_key:
    st.error("Mistral API key not found. Please add it to `.streamlit/secrets.toml`.")

# --- Create Mistral API client ---
@st.cache_resource(show_spinner=False)
def create_openai_client(api_key, base_url):
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

client = create_openai_client(mistral_api_key, mistral_api_base_url)

# --- Cosine Similarity Helper ---
def get_cosine_similarity(source, response, model):
    if model is None:
        return 0.0
    emb1 = model.encode(source, convert_to_tensor=True)
    emb2 = model.encode(response, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
    return round(cosine_sim * 100, 2)

# --- MCQ Generator ---
def generate_mcq_from_pdf(pdf_text, client):
    if not pdf_text or not client:
        st.error("PDF content or API client is missing.")
        return None

    prompt = f"""
You are a quiz master. Based on the content below, generate one multiple choice question with exactly 4 options. Clearly indicate the correct answer. 
Respond in strict JSON format only like this:
{{
  "question": "Question here",
  "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "B"
}}

Content:
\"\"\"
{pdf_text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="mistral-medium",  # Adjust model if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        json_text = response.choices[0].message.content.strip()
        data = json.loads(json_text)

        return {
            "question": data["question"],
            "choices": data["choices"],
            "answer": data["answer"]
        }

    except json.JSONDecodeError:
        st.error("❌ Mistral did not return valid JSON. Possibly missing the 'choices' key.")
        st.code(response.choices[0].message.content)
    except Exception as e:
        st.error(f"🚨 Error generating MCQ: {e}")

    return None

# --- Streamlit UI ---
st.title("📘 AI Quiz Master")
st.write("Upload your study PDF and generate quiz questions powered by Mistral + SentenceTransformer.")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
question_type = st.selectbox("📌 Select question type", ["Multiple Choice"])

# Extract text from PDF
pdf_text = ""
if uploaded_file is not None:
    from PyPDF2 import PdfReader
    reader = PdfReader(uploaded_file)
    pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    st.success("✅ PDF content loaded successfully.")

    if st.button("🧠 Generate Question"):
        if question_type == "Multiple Choice":
            result = generate_mcq_from_pdf(pdf_text, client)
            if result:
                st.subheader("❓ Question")
                st.markdown(result["question"])
                st.subheader("🔘 Options")
                for choice in result["choices"]:
                    st.markdown(f"- {choice}")
                st.markdown(f"✅ **Correct Answer:** {result['answer']}")
