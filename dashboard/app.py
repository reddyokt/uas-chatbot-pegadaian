"""
Dashboard Chatbot Pegadaian (Streamlit)
Jalankan:
  pip install -r requirements.txt
  streamlit run dashboard/app.py
"""

import json
import random
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "pegadaian_dataset.json"
CONFIDENCE_THRESHOLD = 0.35

from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

def preprocess(text: str) -> str:
    text = text.lower()                      # case folding
    text = re.sub(r"[^a-z0-9\s]", " ", text) # hapus tanda baca
    tokens = tokenizer.tokenize(text)        # tokenizing pakai NLTK
    return " ".join(tokens)

def load_dataset(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    tag_to_responses = {item["tag"]: item["responses"] for item in data}

    X, y = [], []
    for item in data:
        for p in item["patterns"]:
            X.append(p)
            y.append(item["tag"])
    return data, X, y, tag_to_responses


@st.cache_resource
def build_model():
    _, X, y, tag_to_responses = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return model, tag_to_responses, acc, labels, cm


def predict_tag(model, text: str):
    probs = model.predict_proba([text])[0]
    best_idx = probs.argmax()
    best_prob = float(probs[best_idx])
    best_tag = model.classes_[best_idx]
    return best_tag, best_prob


st.set_page_config(page_title="Chatbot Pegadaian", page_icon="💬", layout="centered")
st.title("💬 Chatbot Layanan Pegadaian")
st.caption("Metode: TF‑IDF + Multinomial Naive Bayes (Supervised Learning)")

if not DATASET_PATH.exists():
    st.error(f"Dataset tidak ditemukan: {DATASET_PATH}")
    st.stop()

model, tag_to_responses, acc, labels, cm = build_model()

with st.sidebar:
    st.header("Evaluasi Model")
    st.metric("Akurasi (hold‑out)", f"{acc:.2%}")
    st.write("Confusion Matrix (hold‑out):")
    st.dataframe(
        cm,
        use_container_width=True
    )
    st.write("Catatan: dataset kecil → akurasi bisa naik/turun. Tambah variasi patterns akan membantu.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Selamat datang di layanan virtual Pegadaian. Ada yang bisa saya bantu?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Tulis pertanyaan Anda… (mis: 'cara gadai emas')")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tag, conf = predict_tag(model, prompt)
    if conf < CONFIDENCE_THRESHOLD:
        answer = "Maaf, saya belum yakin memahami pertanyaan Anda. Bisa dituliskan dengan cara lain?"
    else:
        answer = random.choice(tag_to_responses.get(tag, ["Maaf, saya belum punya jawabannya."]))
        answer += f"\n\n_(intent: **{tag}**, confidence: **{conf:.2f}**)_"


    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
