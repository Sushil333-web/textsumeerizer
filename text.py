import re
import streamlit as st
import spacy
import docx2txt
import PyPDF2
import joblib
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Contraction mapping
contraction_mapping = {
    "can't": "cannot", "won't": "will not", "i'm": "i am", "ain't": "is not", "it's": "it is", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "i've": "i have", "you're": "you are", "they're": "they are"
    # Add more as needed or import full dictionary
}

def expand_contractions(text):
    pattern = re.compile('({})'.format('|'.join(re.escape(k) for k in contraction_mapping.keys())), flags=re.IGNORECASE)
    def replace(match):
        return contraction_mapping.get(match.group(0).lower(), match.group(0))
    return pattern.sub(replace, text)

def clean_text(text):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[^]]*\]', '', text)  # remove [1], [text], etc.
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text.strip()

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load vectorizer, model, dataset
vectorizer = joblib.load('vectorizer.pkl')
nn_model = joblib.load('nn_model.pkl')
lookup_df = pd.read_csv('amazonFood.csv')

def retrieve_summary(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    _, idx = nn_model.kneighbors(vec)
    return lookup_df.iloc[idx[0][0]]['Summary']

def generate_abstractive_summary(text):
    text_clean = expand_contractions(text)
    doc = nlp(text_clean)
    word_frequencies = {}

    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            word_text = word.text.lower()
            word_frequencies[word_text] = word_frequencies.get(word_text, 0) + 1

    if not word_frequencies:
        return "Text too short to summarize."

    max_freq = max(word_frequencies.values())
    word_frequencies = {word: freq / max_freq for word, freq in word_frequencies.items()}

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            word_text = word.text.lower()
            if word_text in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word_text]

    summarized_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join([sent.text for sent in summarized_sentences])
    return final_summary

def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        if file_type == 'txt':
            return uploaded_file.read().decode("utf-8")
        elif file_type == 'docx':
            return docx2txt.process(uploaded_file)
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return '\n'.join([page.extract_text() for page in pdf_reader.pages])
    return None

# Streamlit UI
st.set_page_config(page_title="Texts Summarization", layout="wide", page_icon="📖")

st.markdown("""
    <style>
        body {background-color: #121212; color: #EAEAEA;}
        .stTextArea textarea {background-color: #1E1E1E; color: #EAEAEA;}
        .stButton>button {background-color: #FF8800; color: white; border-radius: 8px; font-size: 18px;}
        .stSlider>div {color: #FF8800;}
        .stFileUploader {color: #EAEAEA;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #FF8800;'>📖 Texts Summarization</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Developed by: <span style='color: #FF8800;'>Sushil Basnet</span></h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a document (txt, docx, pdf)", type=["txt", "docx", "pdf"])
input_text = st.text_area("✍️ Enter text manually", height=250)

if st.button("✨ Summarize Text"):
    text_to_summarize = extract_text_from_file(uploaded_file) if uploaded_file else input_text.strip()

    if text_to_summarize:
        with st.spinner("⚡ Processing summary..."):
            if len(text_to_summarize.split()) < 100:
                summary = retrieve_summary(text_to_summarize)
                st.success("✅ Summary Retrieved from Dataset!")
            else:
                summary = generate_abstractive_summary(text_to_summarize)
                st.success("✅ Abstractive Summary Generated!")

            st.markdown("### 📝 Summary Output")
            st.write(summary)
    else:
        st.warning("⚠️ Please upload a file or enter some text.")
