import os
import requests

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Set this in GitHub Codespaces secrets
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

def call_mistral(prompt):
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    res = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    return res.json()["choices"][0]["message"]["content"]

def generate_mcq_question(text):
    prompt = f"""
You are a helpful tutor. Based on the following passage:

\"\"\"{text}\"\"\"

Generate:
1. A concise and challenging **multiple-choice question**
2. Four options (A, B, C, D) - one correct, three distractors
3. State which option is correct

Format:
Question: <your question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Answer: <correct letter>
"""
    response = call_mistral(prompt)
    lines = response.strip().split('\n')
    question = lines[0].replace("Question:", "").strip()
    options = [line.split(". ", 1)[1] for line in lines[1:5]]
    correct_line = [line for line in lines if "Answer:" in line][0]
    correct_index = "ABCD".index(correct_line.strip().split(":")[-1].strip())
    return question, options, options[correct_index]

def generate_essay_question(text):
    prompt = f"""
You are an intelligent tutor. From the following passage, generate a challenging **essay-type comprehension question**. Avoid summary prompts.

\"\"\"{text}\"\"\"

Only return the question.
"""
    return call_mistral(prompt).strip()

def evaluate_essay_answer(passage, question, answer):
    prompt = f"""
You are a strict but fair tutor. A student has been asked the question:

\"\"\"{question}\"\"\"

Based on the following text:
\"\"\"{passage}\"\"\"

The student answered:
\"\"\"{answer}\"\"\"

Evaluate the response and give feedback (200 words max). Mention if the answer is accurate, lacks detail, or is incorrect.
"""
    return call_mistral(prompt).strip()
