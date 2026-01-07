# UAS Data Mining / NLP — Chatbot Layanan Pegadaian

## Struktur
- `dataset/pegadaian_dataset.json` : dataset 20 intent (minimal 20 Q&A)
- `chatbot/chatbot.py` : chatbot CLI (terminal) + evaluasi akurasi
- `dashboard/app.py` : dashboard Streamlit (chat UI) + evaluasi

## Cara Menjalankan (VS Code / Terminal)
1) Install dependensi:
```bash
pip install -r requirements.txt
```

2) Jalankan chatbot terminal:
```bash
python chatbot/chatbot.py
```

3) Jalankan dashboard:
```bash
streamlit run dashboard/app.py
```

## Catatan
- Model menggunakan TF-IDF + Multinomial Naive Bayes.
