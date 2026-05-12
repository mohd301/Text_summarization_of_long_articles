import streamlit as st
import joblib
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# -----------------------------
# Load model + vectorizer
# -----------------------------
model = joblib.load("lr_best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
#  preprocess
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(contractions.fix(text.lower()))
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words and len(word)>1
    ]
    return tokens

# -----------------------------
# Summarization function
# -----------------------------
def summarize_article(article, model, vectorizer, max_words=50, max_sentences=3):

    source_sentences = sent_tokenize(article)

    pre_sentences = [" ".join(preprocess(sentence)) for sentence in source_sentences]
    sentence_vectors = vectorizer.transform(pre_sentences)

    scores = model.predict_proba(sentence_vectors)[:, 1]

    ranked_indices = np.argsort(scores)[::-1]

    selected_sentences = []
    total_words = 0

    for i in ranked_indices:
        sentence = source_sentences[i]
        word_count = len(sentence.split())

        selected_sentences.append((i, sentence))
        total_words += word_count

        if (total_words > max_words) or (len(selected_sentences) >= max_sentences):
            break

    selected_sentences = sorted(selected_sentences, key=lambda x: x[0])

    summary = " ".join([s for _, s in selected_sentences])

    return summary

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Extractive Text Summarizer")

st.divider()

st.write("by:")
st.write("Omar Nezar Jaber Jaber -16s2135907")
st.write("MOHAMMED HAMED SAID AL OUFI -56s2197")
st.write("IBRAHIM HAMED JUMA AL-SHAIBANI -16j2124137")

st.divider()

article = st.text_area("Enter your article text here:", height=300)

max_words = st.number_input("Max words", min_value=10, max_value=500, value=50)
st.write("--Note: Summarization will stop AFTER word limit exceeded")
max_sentences = st.number_input("Max sentences", min_value=1, max_value=20, value=3)

if st.button("Summarize"):
    if article.strip():
        summary = summarize_article(
            article,
            model,
            vectorizer,
            max_words=max_words,
            max_sentences=max_sentences
        )

        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")