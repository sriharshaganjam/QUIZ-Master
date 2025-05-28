import streamlit as st
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

question_mode = st.selectbox("Question Type", ["Multiple Choice", "Essay"])
input_text = st.text_area("📘 Paste Educational Content (Text)", height=200)

if st.button("🧠 Generate Question"):
    if not input_text:
        st.warning("Please enter some text.")
    elif not client:
        st.error("Mistral client not initialized.")
    else:
        with st.spinner("Generating question..."):

            prompt = f"""
You are a quiz master. Based on the following text, generate a {"multiple choice" if question_mode == "Multiple Choice" else "short answer"} question in JSON format. 
Ensure the output is ONLY valid JSON.

Text:
"""{input_text}"""

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

Respond with only the JSON, inside a ```json block.
"""

            try:
                response = client.chat.completions.create(
                    model="mistral-tiny",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )

                mistral_raw = response.choices[0].message.content
                question_json = clean_and_parse_json(mistral_raw)

                if not question_json:
                    st.error("❌ Mistral did not return valid JSON. Possibly missing keys.")
                else:
                    st.success("✅ Question Generated!")

                    st.markdown(f"**Question:** {question_json['question']}")

                    if "choices" in question_json:
                        for choice in question_json["choices"]:
                            st.write(f"- {choice}")
                        st.markdown(f"**Answer:** {question_json['answer']}")
                    else:
                        st.markdown(f"**Expected Answer:** {question_json['answer']}")

                    similarity = get_cosine_similarity(input_text, question_json['question'], embedding_model)
                    st.markdown(f"🧠 **Question-Text Similarity:** {similarity}%")

            except Exception as e:
                st.error(f"❌ Error during generation: {e}")
