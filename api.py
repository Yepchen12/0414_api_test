import os
from dotenv import load_dotenv
from langdetect import detect
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define multilingual prompts (for comprehension) + fixed English response instruction
PROMPTS = {
    "en": """You are a media ethics analyst. Please evaluate the following text according to the criteria:
1. Does it contain hate speech?
2. Does it show racial or cultural discrimination?
3. Does it spread misinformation?
4. Does it contain stereotypes or misleading impressions?

Rate each item from 0 to 5 and explain briefly. Then provide an overall score out of 100 and a summary. 
⚠️ Please write the entire response in English.""",
    "zh": """你是一位媒体伦理分析专家。请根据以下标准分析文本：
1. 是否包含仇恨言论？
2. 是否存在种族或文化歧视？
3. 是否传播虚假信息？
4. 是否有刻板印象或引导性误导？

请每项打分（0~5）并说明理由。最后提供总评分（满分100）和简要结论。
⚠️ Please write the entire response in English.""",
    "es": """Eres un analista de ética mediática. Evalúa el texto con base en:
1. ¿Contiene discurso de odio?
2. ¿Muestra discriminación racial o cultural?
3. ¿Difunde información falsa?
4. ¿Incluye estereotipos o impresiones engañosas?

Califica del 0 al 5, explica brevemente cada punto y da un puntaje total de 100. 
⚠️ Please write the entire response in English.""",
    "fr": """Vous êtes un analyste en éthique des médias. Veuillez évaluer ce texte :
1. Contient-il un discours haineux ?
2. Présente-t-il une discrimination raciale ou culturelle ?
3. Diffuse-t-il de fausses informations ?
4. Contient-il des stéréotypes ou impressions trompeuses ?

Attribuez une note de 0 à 5 pour chaque point, puis une note sur 100 et un résumé.
⚠️ Please write the entire response in English.""",
    "ru": """Вы — аналитик в области медиаэтики. Проанализируйте текст:
1. Есть ли язык вражды?
2. Есть ли расовая или культурная дискриминация?
3. Содержит ли дезинформацию?
4. Есть ли стереотипы или вводящие в заблуждение образы?

Оцените каждый пункт по шкале от 0 до 5. Итоговая оценка — из 100. 
⚠️ Please write the entire response in English.""",
    "ar": """أنت محلل أخلاقيات إعلام. قيّم النص وفقًا لما يلي:
1. هل يحتوي على خطاب كراهية؟
2. هل يوجد تمييز عنصري أو ثقافي؟
3. هل ينشر معلومات مضللة؟
4. هل يتضمن صورًا نمطية أو انطباعات مضللة؟

قيّم من 0 إلى 5 لكل نقطة، وقدم التقييم النهائي من 100.
⚠️ Please write the entire response in English."""
}

def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def analyze_text(text, lang):
    prompt = PROMPTS.get(lang, PROMPTS["en"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# Improved score extractor
def extract_scores(result_text):
    lines = result_text.splitlines()
    pattern = re.compile(r"(\d)\s*/\s*5")
    scores = []

    for line in lines:
        match = pattern.search(line)
        if match:
            scores.append(int(match.group(1)))
        if len(scores) == 4:
            break

    # Fill missing with 0s
    while len(scores) < 4:
        scores.append(0)

    return scores

def draw_chart(scores):
    criteria = ["Hate Speech", "Discrimination", "Misinformation", "Stereotyping"]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("Set2")
    sns.barplot(x=criteria, y=scores, palette=colors, ax=ax)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Score (0–5)")
    ax.set_title("Content Harmfulness Evaluation")
    plt.xticks(rotation=20)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# UI
st.set_page_config("Multilingual Harm Analyzer", "📄")
st.title("📄 Multilingual Harmful Content Analyzer")
st.markdown("Upload a **PDF** or **Word** file, or paste text to evaluate content harm based on media ethics principles.")

text_input = st.text_area("Or paste content here:", height=150)
uploaded_file = st.file_uploader("Upload PDF or Word file", type=["pdf", "docx"])

if st.button("🔍 Analyze Content"):
    if not text_input.strip() and not uploaded_file:
        st.error("Please paste some content or upload a document.")
    else:
        with st.spinner("Analyzing, please wait..."):
            if uploaded_file:
                text_input = extract_text(uploaded_file)

            lang = detect(text_input)
            result = analyze_text(text_input, lang)
            scores = extract_scores(result)

            st.success("Analysis complete ✅")
            st.subheader("Results Summary")
            st.markdown(result)
            st.text(f"DEBUG: Extracted Scores: {scores}")

            if all(s == 0 for s in scores):
                st.warning("⚠️ Warning: No scores detected. The result format may have changed.")
            else:
                st.image(draw_chart(scores), caption="Evaluation Summary", use_column_width=True)
