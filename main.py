# main.py

import os
from dotenv import load_dotenv
from langdetect import detect
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import re

# Load env
load_dotenv()

# OpenAI setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt dictionary
PROMPTS = {
    "en": """You are a media ethics analyst...⚠️ Please write the entire response in English.""",
    "zh": """你是一位媒体伦理分析专家...⚠️ Please write the entire response in English.""",
    "es": """Eres un analista de ética mediática...⚠️ Please write the entire response in English.""",
    "fr": """Vous êtes un analyste en éthique des médias...⚠️ Please write the entire response in English.""",
    "ru": """Вы — аналитик в области медиаэтики...⚠️ Please write the entire response in English.""",
    "ar": """أنت محلل أخلاقيات إعلام...⚠️ Please write the entire response in English."""
}

# FastAPI init
app = FastAPI()

class AnalyzeRequest(BaseModel):
    text: str

def analyze_text(text: str, lang: str) -> str:
    prompt = PROMPTS.get(lang, PROMPTS["en"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def extract_scores(text: str) -> list:
    pattern = re.compile(r"(\d)\s*/\s*5")
    lines = text.splitlines()
    scores = []
    for line in lines:
        match = pattern.search(line)
        if match:
            scores.append(int(match.group(1)))
        if len(scores) == 4:
            break
    while len(scores) < 4:
        scores.append(0)
    return scores

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    lang = detect(request.text)
    analysis = analyze_text(request.text, lang)
    scores = extract_scores(analysis)
    return {
        "language": lang,
        "scores": {
            "hate_speech": scores[0],
            "discrimination": scores[1],
            "misinformation": scores[2],
            "stereotyping": scores[3],
        },
        "raw_analysis": analysis
    }
