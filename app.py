import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import json
import re
import random
import hashlib
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(page_title="AI Quiz Master", layout="centered", page_icon="📚")

st.title("📚 AI Quiz Master For Students To Aid Subject Mastery")
st.markdown("Auto generate intelligent questions to test your knowledge about any PDF using advanced AI techniques & obtain personalized feedback on your submitted answers.")

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

class ContentAnalyzer:
    """Advanced content analysis and segmentation"""
    
    def __init__(self, text):
        self.full_text = text
        self.sections = []
        self.key_concepts = []
        self.embeddings = None
        self.analyzed = False
    
    def analyze_content(self, embedding_model=None):
        """Perform comprehensive content analysis"""
        if self.analyzed:
            return
        
        # Split into logical sections
        self.sections = self._split_into_sections()
        
        # Extract key concepts
        self.key_concepts = self._extract_key_concepts()
        
        # Generate embeddings for semantic search
        if embedding_model:
            self._generate_embeddings(embedding_model)
        
        self.analyzed = True
    
    def _split_into_sections(self):
        """Split content into logical sections based on structure"""
        sections = []
        
        # Split by page breaks first
        pages = self.full_text.split("--- Page")
        
        for i, page_content in enumerate(pages):
            if not page_content.strip():
                continue
                
            # Further split by paragraphs and headers
            paragraphs = [p.strip() for p in page_content.split('\n\n') if p.strip()]
            
            # Group paragraphs into sections (3-5 paragraphs per section)
            for j in range(0, len(paragraphs), random.randint(3, 5)):
                section_text = '\n\n'.join(paragraphs[j:j+5])
                if len(section_text) > 200:  # Only include substantial sections
                    sections.append({
                        'content': section_text,
                        'page': i,
                        'section_id': len(sections),
                        'word_count': len(section_text.split()),
                        'key_sentences': self._extract_key_sentences(section_text)
                    })
        
        return sections
    
    def _extract_key_sentences(self, text):
        """Extract key sentences from a text section"""
        try:
            sentences = sent_tokenize(text)
            # Return sentences that are neither too short nor too long
            key_sentences = [s for s in sentences if 20 <= len(s.split()) <= 50]
            return key_sentences[:3]  # Top 3 key sentences
        except:
            # Fallback if NLTK fails
            sentences = text.split('. ')
            return [s for s in sentences[:3] if len(s.split()) >= 10]
    
    def _extract_key_concepts(self):
        """Extract key concepts and topics from the content"""
        try:
            words = word_tokenize(self.full_text.lower())
            stop_words = set(stopwords.words('english'))
            
            # Remove stopwords and short words
            meaningful_words = [w for w in words if w.isalpha() and len(w) > 3 and w not in stop_words]
            
            # Count frequency
            word_freq = defaultdict(int)
            for word in meaningful_words:
                word_freq[word] += 1
            
            # Get top concepts
            top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            return [concept[0] for concept in top_concepts]
        except:
            # Simple fallback
            words = self.full_text.split()
            return list(set([w.lower() for w in words if len(w) > 5]))[:20]
    
    def _generate_embeddings(self, model):
        """Generate embeddings for semantic similarity"""
        try:
            section_texts = [section['content'] for section in self.sections]
            self.embeddings = model.encode(section_texts)
        except Exception as e:
            st.warning(f"Could not generate embeddings: {e}")
    
    def get_diverse_content(self, used_sections=None, strategy='random'):
        """Get content using different strategies for diversity"""
        if not self.sections:
            return self.full_text[:2000]
        
        available_sections = self.sections.copy()
        
        # Remove already used sections
        if used_sections:
            available_sections = [s for s in available_sections if s['section_id'] not in used_sections]
        
        if not available_sections:
            # If all sections used, reset and use semantic diversity
            available_sections = self.sections.copy()
            strategy = 'semantic'
        
        if strategy == 'random':
            section = random.choice(available_sections)
        elif strategy == 'longest':
            section = max(available_sections, key=lambda x: x['word_count'])
        elif strategy == 'concept_rich':
            # Choose section with most key concepts
            section = max(available_sections, key=lambda x: 
                        sum(1 for concept in self.key_concepts if concept in x['content'].lower()))
        elif strategy == 'semantic':
            # Use embedding similarity to find diverse content
            if self.embeddings is not None and used_sections:
                # Find section most different from used ones
                section = self._find_most_different_section(available_sections, used_sections)
            else:
                section = random.choice(available_sections)
        else:
            section = random.choice(available_sections)
        
        return section['content'], section['section_id']
    
    def _find_most_different_section(self, available_sections, used_sections):
        """Find section most semantically different from used ones"""
        try:
            if not self.embeddings:
                return random.choice(available_sections)
            
            max_diff = -1
            best_section = available_sections[0]
            
            for section in available_sections:
                section_embedding = self.embeddings[section['section_id']]
                
                # Calculate average similarity to used sections
                similarities = []
                for used_id in used_sections:
                    if used_id < len(self.embeddings):
                        used_embedding = self.embeddings[used_id]
                        sim = util.pytorch_cos_sim(section_embedding, used_embedding).item()
                        similarities.append(sim)
                
                # Lower average similarity = more different
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                
                if avg_similarity < max_diff or max_diff == -1:
                    max_diff = avg_similarity
                    best_section = section
            
            return best_section
        except:
            return random.choice(available_sections)

class QuestionGenerator:
    """Advanced question generation with diversity tracking"""
    
    def __init__(self):
        self.question_types = [
            'factual', 'analytical', 'comparative', 'application', 
            'synthesis', 'evaluation', 'definition', 'cause_effect'
        ]
        self.difficulty_levels = ['basic', 'intermediate', 'advanced']
        self.generated_questions = []
        self.used_sections = set()
    
    def generate_diverse_question(self, content_analyzer, client, question_format='Multiple Choice'):
        """Generate a diverse question using advanced strategies"""
        
        # Choose diversity strategy
        strategies = ['random', 'concept_rich', 'longest', 'semantic']
        weights = [0.4, 0.3, 0.2, 0.1]  # Prefer random and concept-rich
        strategy = random.choices(strategies, weights=weights)[0]
        
        # Get diverse content
        content, section_id = content_analyzer.get_diverse_content(
            used_sections=self.used_sections, 
            strategy=strategy
        )
        
        # Track used sections
        self.used_sections.add(section_id)
        
        # Choose question type and difficulty
        question_type = random.choice(self.question_types)
        difficulty = random.choice(self.difficulty_levels)
        
        # Generate question
        question_data = self._generate_specific_question(
            content, question_type, difficulty, question_format, client
        )
        
        if question_data:
            # Add metadata
            question_data['metadata'] = {
                'section_id': section_id,
                'question_type': question_type,
                'difficulty': difficulty,
                'strategy': strategy,
                'content_preview': content[:100] + "..."
            }
            self.generated_questions.append(question_data)
        
        return question_data
    
    def _generate_specific_question(self, content, q_type, difficulty, format_type, client):
        """Generate question with specific type and difficulty"""
        
        # Question type specific prompts
        type_prompts = {
            'factual': "Create a question that tests recall of specific facts or information.",
            'analytical': "Create a question that requires breaking down and analyzing information.",
            'comparative': "Create a question comparing different concepts or ideas.",
            'application': "Create a question about applying knowledge to new situations.",
            'synthesis': "Create a question that combines multiple concepts or ideas.",
            'evaluation': "Create a question that requires making judgments or assessments.",
            'definition': "Create a question about defining or explaining key terms.",
            'cause_effect': "Create a question about cause-and-effect relationships."
        }
        
        difficulty_instructions = {
            'basic': "Make this a basic level question testing fundamental understanding.",
            'intermediate': "Make this an intermediate level question requiring some analysis.",
            'advanced': "Make this an advanced level question requiring deep critical thinking."
        }
        
        if format_type == "Multiple Choice":
            prompt = f"""
Create a {difficulty} level {q_type} multiple choice question based on the content below.

Instructions:
- {type_prompts.get(q_type, 'Create a thoughtful question.')}
- {difficulty_instructions.get(difficulty, '')}
- Make the question unique and avoid repetitive patterns
- Ensure all options are plausible but only one is clearly correct
- Use varied question structures and phrasings

Return ONLY a valid JSON object:
{{
  "question": "Your unique question here",
  "choices": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
  "answer": "A"
}}

Content:
{content[:1800]}
"""
        else:  # Essay question
            prompt = f"""
Create a {difficulty} level {q_type} essay question based on the content below.

Instructions:
- {type_prompts.get(q_type, 'Create a thoughtful question.')}
- {difficulty_instructions.get(difficulty, '')}
- Make the question thought-provoking and unique
- Require comprehensive understanding, not just memorization

Return ONLY a valid JSON object:
{{
  "question": "Your unique essay question here",
  "answer": "Comprehensive key points and concepts that should be addressed"
}}

Content:
{content[:1800]}
"""

        try:
            response = client.chat.completions.create(
                model="mistral-small",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Higher temperature for more creativity
                max_tokens=600
            )
            
            raw_response = response.choices[0].message.content
            return self._clean_and_parse_json(raw_response)
            
        except Exception as e:
            st.error(f"Error generating question: {e}")
            return None
    
    def _clean_and_parse_json(self, raw_text):
        """Parse JSON from response with robust error handling"""
        try:
            # Remove markdown code blocks
            json_text = re.sub(r'```json\s*', '', raw_text)
            json_text = re.sub(r'```\s*$', '', json_text)
            
            # Find JSON object
            json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            
            json_text = json_text.strip()
            
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return self._fix_and_parse_json(json_text)
                
        except Exception as e:
            st.warning(f"JSON parse error: {e}")
            return self._extract_Youtube_fallback(raw_text)
    
    def _fix_and_parse_json(self, json_text):
        """Fix common JSON issues"""
        try:
            # Extract components with regex
            question_match = re.search(r'"question"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', json_text)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', json_text)
            choices_match = re.search(r'"choices"\s*:\s*\[(.*?)\]', json_text, re.DOTALL)
            
            if question_match and answer_match:
                result = {
                    "question": question_match.group(1),
                    "answer": answer_match.group(1)
                }
                
                if choices_match:
                    choices_str = choices_match.group(1)
                    choice_matches = re.findall(r'"([^"]*(?:\\.[^"]*)*)"', choices_str)
                    if choice_matches:
                        result["choices"] = choice_matches
                
                return result
        except:
            pass
        return None
    
    def _extract_Youtube_fallback(self, raw_text):
        """Fallback extraction method"""
        try:
            lines = raw_text.split('\n')
            question = None
            answer = None
            choices = []
            
            for line in lines:
                line = line.strip()
                
                if 'question' in line.lower() and not question:
                    patterns = [
                        r'"question"\s*:\s*"([^"]+)"',
                        r'question:\s*"([^"]+)"',
                        r'question:\s*([^,\n]+)',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            question = match.group(1).strip()
                            break
                
                elif 'answer' in line.lower() and not answer:
                    patterns = [
                        r'"answer"\s*:\s*"([^"]+)"',
                        r'answer:\s*"([^"]+)"',
                        r'answer:\s*([^,\n]+)',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            answer = match.group(1).strip()
                            break
                
                elif any(letter in line for letter in ['A.', 'B.', 'C.', 'D.']):
                    match = re.search(r'([A-D]\.[^"]*)', line)
                    if match:
                        choices.append(match.group(1).strip())
            
            if question and answer:
                result = {"question": question, "answer": answer}
                if choices:
                    result["choices"] = choices
                return result
        except:
            pass
        return None
    
    def get_question_stats(self):
        """Get statistics about generated questions"""
        if not self.generated_questions:
            return {}
        
        stats = {
            'total_questions': len(self.generated_questions),
            'unique_sections': len(self.used_sections),
            'question_types': defaultdict(int),
            'difficulty_levels': defaultdict(int),
            'strategies_used': defaultdict(int)
        }
        
        for q in self.generated_questions:
            if 'metadata' in q:
                metadata = q['metadata']
                stats['question_types'][metadata.get('question_type', 'unknown')] += 1
                stats['difficulty_levels'][metadata.get('difficulty', 'unknown')] += 1
                stats['strategies_used'][metadata.get('strategy', 'unknown')] += 1
        
        return stats
    
    def reset_diversity_tracking(self):
        """Reset tracking for new document or when requested"""
        self.used_sections.clear()
        self.generated_questions.clear()

def extract_text_from_pdf(uploaded_file):
    """Extract text content from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page_text
        
        return text_content.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
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

def evaluate_essay_answer(user_answer, expected_answer, question, model):
    """Evaluate essay answer and provide detailed feedback"""
    if not user_answer.strip():
        return 0, "Please provide an answer."
    
    similarity_score = get_cosine_similarity(expected_answer, user_answer, model)
    
    feedback_prompt = f"""
Compare the student answer with the expected answer and provide constructive feedback.

Question: {question}
Expected Answer: {expected_answer}
Student Answer: {user_answer}

Provide feedback in this format:
- Strengths: What the student got right
- Areas for improvement: What could be better or missing
- Suggestions: Specific recommendations

Do not include any score in your response."""
    
    try:
        response = client.chat.completions.create(
            model="mistral-small",
            messages=[{"role": "user", "content": feedback_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        detailed_feedback = response.choices[0].message.content
        return similarity_score, detailed_feedback
        
    except Exception as e:
        if similarity_score > 80:
            feedback = "Excellent! Your answer covers the key points well."
        elif similarity_score > 60:
            feedback = "Good answer, but some important points might be missing."
        elif similarity_score > 40:
            feedback = "Your answer shows understanding but lacks several key concepts."
        else:
            feedback = "Your answer needs significant improvement. Focus on main concepts."
        
        return similarity_score, feedback

# Initialize session state
session_keys = [
    'current_question', 'current_answer', 'current_choices', 'pdf_content',
    'user_answer_mc', 'user_answer_essay', 'show_feedback', 'current_question_type',
    'content_analyzer', 'question_generator', 'question_metadata', 'uploaded_file_name' 
]

for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

if 'user_answer_essay' not in st.session_state:
    st.session_state.user_answer_essay = ""

# Ensure question_generator is initialized before use
if 'question_generator' not in st.session_state or st.session_state.question_generator is None:
    st.session_state.question_generator = QuestionGenerator()

# Main interface
st.markdown("### 📁 Upload Learning Material")

uploaded_file = st.file_uploader(
    "Upload a PDF file containing educational content",
    type=["pdf"],
    help="The entire PDF will be analyzed for maximum question diversity"
)

# Process uploaded file
if uploaded_file is not None:
    # Check if a new file is uploaded or if content_analyzer needs re-initialization
    if st.session_state.get('uploaded_file_name') != uploaded_file.name or st.session_state.content_analyzer is None:
        with st.spinner("🔍 Analyzing PDF content comprehensively..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.session_state.pdf_content = extracted_text
                st.session_state.uploaded_file_name = uploaded_file.name
                
                # Initialize content analyzer
                st.session_state.content_analyzer = ContentAnalyzer(extracted_text)
                st.session_state.content_analyzer.analyze_content(embedding_model)
                
                # Reset question generator for new document
                # Ensure question_generator is an instance before calling its method
                if st.session_state.question_generator is None:
                    st.session_state.question_generator = QuestionGenerator()
                st.session_state.question_generator.reset_diversity_tracking()
                
                st.success(f"✅ Successfully analyzed {uploaded_file.name}")
                
                # Show analysis stats
                analyzer = st.session_state.content_analyzer
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📄 Total Characters", f"{len(extracted_text):,}")
                with col2:
                    st.metric("📑 Sections Found", len(analyzer.sections))
                with col3:
                    st.metric("🎯 Key Concepts", len(analyzer.key_concepts))
                with col4:
                    st.metric("🧠 Analysis Status", "✅ Complete")
                
                # Show preview
                with st.expander("📖 Content Analysis Preview"):
                    st.subheader("Key Concepts Identified:")
                    st.write(", ".join(analyzer.key_concepts[:15]))
                    
                    st.subheader("Sample Content:")
                    st.text(extracted_text[:800] + "..." if len(extracted_text) > 800 else extracted_text)
            else:
                st.error("Failed to extract text from the PDF. Please try another file.")

# Enhanced question generation section
if st.session_state.pdf_content and st.session_state.content_analyzer:
    st.markdown("### 🧠 Generate Diverse Quiz Questions")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        question_type = st.selectbox(
            "Select question format:",
            ["Multiple Choice", "Essay"],
            help="Choose the format for your questions"
        )
    
    with col2:
        generate_btn = st.button(
            "🎲 Generate Question", 
            type="primary",
            use_container_width=True
        )
    
    with col3:
        reset_diversity = st.button(
            "🔄 Reset Diversity",
            help="Reset question tracking to allow revisiting all content",
            use_container_width=True
        )
    
    if reset_diversity:
        # Ensure question_generator is an instance before calling its method
        if st.session_state.question_generator is None:
            st.session_state.question_generator = QuestionGenerator()
        st.session_state.question_generator.reset_diversity_tracking()
        st.success("✅ Diversity tracking reset! All content sections are now available again.")
    
    # Show question generation stats
    # Ensure question_generator is an instance before calling its method
    if st.session_state.question_generator is None:
        st.session_state.question_generator = QuestionGenerator()
    stats = st.session_state.question_generator.get_question_stats()
    if stats.get('total_questions', 0) > 0:
        with st.expander("📊 Question Generation Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Question Types Generated:**")
                for q_type, count in stats['question_types'].items():
                    st.write(f"- {q_type.title()}: {count}")
                
                st.write("**Difficulty Levels:**")
                for level, count in stats['difficulty_levels'].items():
                    st.write(f"- {level.title()}: {count}")
            
            with col2:
                st.write("**Content Strategies Used:**")
                for strategy, count in stats['strategies_used'].items():
                    st.write(f"- {strategy.title()}: {count}")
                
                st.metric("Sections Covered", f"{stats['unique_sections']}/{len(st.session_state.content_analyzer.sections)}")
    
    if generate_btn:
        # Reset previous state
        st.session_state.show_feedback = False
        st.session_state.user_answer_mc = None
        st.session_state.user_answer_essay = ""
        st.session_state.current_question_type = question_type
        
        with st.spinner("🤖 Generating diverse question using advanced AI..."):
            # Ensure question_generator is an instance before calling its method
            if st.session_state.question_generator is None:
                st.session_state.question_generator = QuestionGenerator()
            question_data = st.session_state.question_generator.generate_diverse_question(
                st.session_state.content_analyzer,
                client,
                question_type
            )
            
            if question_data:
                st.session_state.current_question = question_data.get('question')
                st.session_state.current_answer = question_data.get('answer')
                st.session_state.current_choices = question_data.get('choices')
                st.session_state.question_metadata = question_data.get('metadata', {})
                
                st.success("✅ Unique question generated! Answer below.")
                
                # Show question metadata
                if st.session_state.question_metadata:
                    metadata = st.session_state.question_metadata
                    st.info(
                        f"**Question Type:** {metadata.get('question_type', 'N/A').title()} | "
                        f"**Difficulty:** {metadata.get('difficulty', 'N/A').title()} | "
                        f"**Strategy:** {metadata.get('strategy', 'N/A').title()}"
                    )
            else:
                st.error("❌ Failed to generate question. Please try again.")

# Display question and answer interface (rest remains the same)
if st.session_state.current_question:
    st.markdown("### 📝 Answer the Question")
    
    st.markdown(f"**Question:** {st.session_state.current_question}")
    
    if st.session_state.current_question_type == "Multiple Choice" and st.session_state.current_choices:
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
        
        word_count = len(st.session_state.user_answer_essay.split())
        st.caption(f"Word count: {word_count}/100")
        
        answer_provided = bool(st.session_state.user_answer_essay.strip())
    
    # Evaluate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("✅ Evaluate Answer", type="primary", disabled=not answer_provided):
            st.session_state.show_feedback = True

# Display feedback (rest of the feedback logic remains the same)
if st.session_state.show_feedback:
    st.markdown("### 📊 Evaluation Results")
    
    if st.session_state.current_question_type == "Multiple Choice" and st.session_state.current_choices:
        if st.session_state.user_answer_mc:
            selected_letter = st.session_state.user_answer_mc.split('.')[0]
            correct_answer = st.session_state.current_answer
            
            if selected_letter == correct_answer:
                st.success("🎉 Correct! Well done!")
                st.balloons()
            else:
                st.error(f"❌ Incorrect. The correct answer was: **{correct_answer}**")
            
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
            
            if score > 80:
                st.success(f"📈 **Score: {score}%** - Excellent!")
            elif score > 60:
                st.warning(f"📈 **Score: {score}%** - Good")
            else:
                st.error(f"📈 **Score: {score}%** - Needs Improvement")
            
            st.markdown("**Detailed Feedback:**")
            st.info(feedback)
            
            with st.expander("📚 View Expected Answer Points"):
                st.markdown(f"**Key points to cover:** {st.session_state.current_answer}")
    
    # Show question metadata and relevance
    if st.session_state.question_metadata:
        metadata = st.session_state.question_metadata
        with st.expander("🔍 Question Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Section Used:** {metadata.get('section_id', 'N/A')}")
                st.write(f"**Question Type:** {metadata.get('question_type', 'N/A').title()}")
                st.write(f"**Difficulty Level:** {metadata.get('difficulty', 'N/A').title()}")
            with col2:
                st.write(f"**Selection Strategy:** {metadata.get('strategy', 'N/A').title()}")
                if 'content_preview' in metadata:
                    st.write(f"**Content Sample:** {metadata['content_preview']}")
    
    # Calculate question relevance to full content
    if embedding_model and st.session_state.pdf_content:
        relevance_score = get_cosine_similarity(
            st.session_state.pdf_content[:1000],
            st.session_state.current_question,
            embedding_model
        )
        st.caption(f"🎯 Question relevance to document: {relevance_score}%")

# Footer with enhanced information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📚 Enhanced AI Quiz Master v2.0<br>
        🚀 Features: Advanced Content Analysis • Semantic Diversity • Question Type Variety • Difficulty Levels<br>
        🤖 Powered by Mistral AI, Sentence Transformers & NLTK
    </div>
    """,
    unsafe_allow_html=True
)
