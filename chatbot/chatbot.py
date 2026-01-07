"""
Chatbot Layanan Pegadaian (UAS Data Mining / NLP)
Metode: TF-IDF + Multinomial Naive Bayes (Supervised Learning)
Jalankan: python chatbot.py
"""

import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "pegadaian_dataset.json"
CONFIDENCE_THRESHOLD = 0.35  # jika terlalu rendah -> fallback


def preprocess(text: str) -> str:
    """Case folding + hapus tanda baca + rapikan spasi."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    tag_to_responses = {item["tag"]: item["responses"] for item in data}

    X, y = [], []
    for item in data:
        for p in item["patterns"]:
            X.append(p)
            y.append(item["tag"])
    return X, y, tag_to_responses


def train_model(X, y):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    model.fit(X, y)
    return model


def evaluate_quick(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Evaluasi Singkat (Hold-out Test) ===")
    print(f"Akurasi: {acc:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=== Selesai ===\n")


def predict_tag(model, text: str):
    # Naive Bayes mendukung predict_proba
    probs = model.predict_proba([text])[0]
    best_idx = probs.argmax()
    best_prob = probs[best_idx]
    best_tag = model.classes_[best_idx]
    return best_tag, float(best_prob)


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    X, y, tag_to_responses = load_dataset(DATASET_PATH)

    # split untuk evaluasi sederhana (agar sesuai rubrik akurasi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_quick(model, X_test, y_test)

    print("Chatbot Pegadaian siap. Ketik 'exit' untuk keluar.\n")
    while True:
        user = input("Anda: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit", "keluar"}:
            print("Chatbot: Terima kasih telah menggunakan layanan Pegadaian. Sampai jumpa!")
            break

        tag, conf = predict_tag(model, user)
        if conf < CONFIDENCE_THRESHOLD:
            print("Chatbot: Maaf, saya belum yakin memahami pertanyaan Anda. Bisa dituliskan dengan cara lain?")
            continue

        response = random.choice(tag_to_responses.get(tag, ["Maaf, saya belum punya jawabannya."]))
        print(f"Chatbot: {response}  (intent={tag}, confidence={conf:.2f})")


if __name__ == "__main__":
    main()
