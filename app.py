import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import json
import re
import random

st.set_page_config(page_title="AI Quiz Master", layout="centered", page_icon="📚")

st.title("📚 AI Quiz Master")
st.markdown("Generate intelligent questions from any PDF content using Mistral AI + Sentence Transformers.")

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    try:
        with st.spinner("Loading AI models..."):
            return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        return None

# Load the embedding model
embedding_model = load_embedding_model()

# Get API configuration
mistral_api_key = st.secrets.get("mistral", {}).get("api_key")
mistral_api_base_url = st.secrets.get("mistral", {}).get("base_url", "https://api.mistral.ai/v1")

if not mistral_api_key:
    st.error("❌ Mistral API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

@st.cache_resource(show_spinner=False)
def create_openai_client(api_key, base_url):
    """Create and cache OpenAI client for Mistral API"""
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

client = create_openai_client(mistral_api_key, mistral_api_base_url)

def extract_text_from_pdf(uploaded_file):
    """Extract text content from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page_text
        
        return text_content.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def get_random_text_chunk(text, chunk_size=2000):
    """Get a random chunk of text for question generation"""
    if len(text) <= chunk_size:
        return text
    
    # Split by paragraphs to avoid cutting sentences
    paragraphs = text.split('\n\n')
    
    # If we have many paragraphs, select a random subset
    if len(paragraphs) > 5:
        start_idx = random.randint(0, max(0, len(paragraphs) - 5))
        selected_paragraphs = paragraphs[start_idx:start_idx + 5]
        chunk = '\n\n'.join(selected_paragraphs)
        
        # If chunk is still too long, truncate
        if len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]
            # Find last complete sentence
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.8:  # Only truncate if we don't lose too much
                chunk = chunk[:last_period + 1]
        
        return chunk
    else:
        return text[:chunk_size] if len(text) > chunk_size else text

def clean_and_parse_json(raw_text):
    """Parse JSON from Mistral response, handling various formats"""
    try:
        # Try to find JSON within code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            return json.loads(json_text)
        
        # Try to find JSON without code blocks
        json_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            return json.loads(json_text)
        
        # Try parsing the entire response as JSON
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ JSON parse error: {e}")
        st.text("Raw response:")
        st.code(raw_text)
        return None

def get_cosine_similarity(text1, text2, model):
    """Calculate cosine similarity between two texts"""
    if model is None:
        return 0.0
    try:
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
        return round(cosine_sim * 100, 2)
    except Exception as e:
        st.warning(f"Error calculating similarity: {e}")
        return 0.0

def generate_question(content, question_type):
    """Generate question using Mistral API"""
    
    # Use a chunk of content for question generation
    content_chunk = get_random_text_chunk(content, 2000)
    
    if question_type == "Multiple Choice":
        prompt = f"""
You are an expert quiz creator. Based on the following educational content, create ONE multiple choice question with 4 options.

IMPORTANT: Your response must be ONLY valid JSON in this exact format:

```json
{{
  "question": "Your question here?",
  "choices": ["A. First option", "B. Second option", "C. Third option", "D. Fourth option"],
  "answer": "A"
}}
```

Content:
{content_chunk}

Requirements:
- Make the question challenging but fair
- Ensure only one option is clearly correct
- Make incorrect options plausible but distinctly wrong
- Base the question on key concepts from the content
- Keep question and options concise but clear
"""
    else:  # Essay question
        prompt = f"""
You are an expert quiz creator. Based on the following educational content, create ONE essay question that requires understanding and explanation.

IMPORTANT: Your response must be ONLY valid JSON in this exact format:

```json
{{
  "question": "Your essay question here?",
  "answer": "Key points that should be covered in a good answer: point 1, point 2, point 3, etc."
}}
```

Content:
{content_chunk}

Requirements:
- Create a question that requires explanation or analysis
- The answer should list 3-5 key points that demonstrate understanding
- Focus on concepts that require critical thinking
- Keep the question clear and specific
"""

    try:
        response = client.chat.completions.create(
            model="mistral-tiny",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        raw_response = response.choices[0].message.content
        return clean_and_parse_json(raw_response)
        
    except Exception as e:
        st.error(f"Error generating question: {e}")
        return None

def evaluate_essay_answer(user_answer, expected_answer, question, model):
    """Evaluate essay answer and provide detailed feedback"""
    if not user_answer.strip():
        return 0, "Please provide an answer."
    
    # Calculate similarity score
    similarity_score = get_cosine_similarity(expected_answer, user_answer, model)
    
    # Generate detailed feedback using Mistral
    feedback_prompt = f"""
Compare the following student answer with the expected answer and provide constructive feedback.

Question: {question}

Expected Answer: {expected_answer}

Student Answer: {user_answer}

Provide feedback in this format:
- Score: {similarity_score}%
- Strengths: What the student got right
- Areas for improvement: What could be better or what's missing
- Suggestions: Specific recommendations
"""
    
    try:
        response = client.chat.completions.create(
            model="mistral-tiny",
            messages=[{"role": "user", "content": feedback_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        detailed_feedback = response.choices[0].message.content
        return similarity_score, detailed_feedback
        
    except Exception as e:
        # Fallback to basic feedback if API call fails
        if similarity_score > 80:
            feedback = "Excellent! Your answer covers the key points well."
        elif similarity_score > 60:
            feedback = "Good answer, but some important points might be missing. Review the expected answer."
        elif similarity_score > 40:
            feedback = "Your answer shows some understanding, but lacks several key concepts."
        else:
            feedback = "Your answer needs significant improvement. Focus on the main concepts from the content."
        
        return similarity_score, feedback

# Initialize session state
for key in ['current_question', 'current_answer', 'current_choices', 'pdf_content', 
            'user_answer_mc', 'user_answer_essay', 'show_feedback', 'question_type']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'user_answer_essay' not in st.session_state:
    st.session_state.user_answer_essay = ""

# Main interface
st.markdown("### 📁 Upload Learning Material")

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF file containing educational content",
    type=["pdf"],
    help="Supported format: PDF files only"
)

# Process uploaded file
if uploaded_file is not None:
    if st.session_state.get('uploaded_file_name') != uploaded_file.name:
        # New file uploaded
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.session_state.pdf_content = extracted_text
                st.session_state.uploaded_file_name = uploaded_file.name
                st.success(f"✅ Successfully extracted text from {uploaded_file.name}")
                st.info(f"📄 Document contains {len(extracted_text)} characters")
                
                # Show preview
                with st.expander("📖 Preview extracted content"):
                    st.text(extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text)
            else:
                st.error("Failed to extract text from the PDF. Please try another file.")

# Question generation section
if st.session_state.pdf_content:
    st.markdown("### 🧠 Generate Quiz Questions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question_type = st.selectbox(
            "Select question type:",
            ["Multiple Choice", "Essay"],
            help="Choose the type of question you want to practice"
        )
    
    with col2:
        generate_btn = st.button(
            "🎲 Generate Question", 
            type="primary",
            use_container_width=True
        )
    
    if generate_btn:
        # Reset previous state
        st.session_state.show_feedback = False
        st.session_state.user_answer_mc = None
        st.session_state.user_answer_essay = ""
        st.session_state.question_type = question_type
        
        with st.spinner("🤖 Generating question..."):
            question_data = generate_question(st.session_state.pdf_content, question_type)
            
            if question_data:
                st.session_state.current_question = question_data.get('question')
                st.session_state.current_answer = question_data.get('answer')
                st.session_state.current_choices = question_data.get('choices')
                st.success("✅ Question generated! Answer below.")
            else:
                st.error("❌ Failed to generate question. Please try again.")

# Display question and answer interface
if st.session_state.current_question:
    st.markdown("### 📝 Answer the Question")
    
    # Display question
    st.markdown(f"**Question:** {st.session_state.current_question}")
    
    # Answer input based on question type
    if st.session_state.current_choices:  # Multiple Choice
        st.session_state.user_answer_mc = st.radio(
            "Select your answer:",
            st.session_state.current_choices,
            index=None,
            key="mc_radio"
        )
        
        answer_provided = st.session_state.user_answer_mc is not None
        
    else:  # Essay
        st.session_state.user_answer_essay = st.text_area(
            "Write your answer (maximum 100 words):",
            value=st.session_state.user_answer_essay,
            max_chars=700,
            height=150,
            help="Provide a comprehensive answer covering the key points",
            key="essay_textarea"
        )
        
        # Word count
        word_count = len(st.session_state.user_answer_essay.split())
        st.caption(f"Word count: {word_count}/100")
        
        answer_provided = bool(st.session_state.user_answer_essay.strip())
    
    # Evaluate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✅ Evaluate Answer", type="primary", disabled=not answer_provided):
            st.session_state.show_feedback = True

# Display feedback
if st.session_state.show_feedback:
    st.markdown("### 📊 Evaluation Results")
    
    if st.session_state.current_choices:  # Multiple Choice
        if st.session_state.user_answer_mc:
            # Extract letter from choice
            selected_letter = st.session_state.user_answer_mc.split('.')[0]
            correct_answer = st.session_state.current_answer
            
            if selected_letter == correct_answer:
                st.success("🎉 Correct! Well done!")
                st.balloons()
            else:
                st.error(f"❌ Incorrect. The correct answer was: **{correct_answer}**")
            
            # Show all options with correct answer highlighted
            st.markdown("**Answer Review:**")
            for choice in st.session_state.current_choices:
                choice_letter = choice.split('.')[0]
                if choice_letter == correct_answer:
                    st.markdown(f"✅ **{choice}** ← Correct Answer")
                elif choice_letter == selected_letter and selected_letter != correct_answer:
                    st.markdown(f"❌ {choice} ← Your Answer")
                else:
                    st.markdown(f"   {choice}")
    
    else:  # Essay
        if st.session_state.user_answer_essay.strip():
            with st.spinner("Evaluating your essay answer..."):
                score, feedback = evaluate_essay_answer(
                    st.session_state.user_answer_essay,
                    st.session_state.current_answer,
                    st.session_state.current_question,
                    embedding_model
                )
            
            # Display score with color coding
            if score > 80:
                st.success(f"📈 **Score: {score}%** - Excellent!")
            elif score > 60:
                st.warning(f"📈 **Score: {score}%** - Good")
            else:
                st.error(f"📈 **Score: {score}%** - Needs Improvement")
            
            # Display detailed feedback
            st.markdown("**Detailed Feedback:**")
            st.info(feedback)
            
            # Show expected answer
            with st.expander("📚 View Expected Answer Points"):
                st.markdown(f"**Key points to cover:** {st.session_state.current_answer}")
    
    # Calculate question relevance
    if embedding_model and st.session_state.pdf_content:
        relevance_score = get_cosine_similarity(
            st.session_state.pdf_content[:1000],  # First 1000 chars for efficiency
            st.session_state.current_question,
            embedding_model
        )
        st.caption(f"🎯 Question relevance to content: {relevance_score}%")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📚 AI Quiz Master - Powered by Mistral AI & Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True
)
