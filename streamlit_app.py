import streamlit as st
import gensim
from gensim import corpora
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np
import re
import emoji
import torch
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =============================================================================
# 1. Setup & Configuration
# =============================================================================

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# =============================================================================
# 2. Preprocessing Resources
# =============================================================================

# --- Stopwords ---
STOPWORD_PATH = "stopwordbahasa.txt"
additional_stopwords = []
if os.path.exists(STOPWORD_PATH):
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        additional_stopwords = [line.strip() for line in f.readlines()]
    additional_stopwords = [sw for sw in additional_stopwords if sw]

stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'nya', 'dana']

stop_words = set(stop_factory.get_stop_words())
stop_words.update(more_stopword)
stop_words.update(additional_stopwords)

# --- Stemmer ---
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# --- Normalization Dictionary ---
normalization_dict = {
    'ae': 'saja','aja': 'saja','ajah': 'saja','aj': 'saja','jha': 'saja','sj': 'saja',
    'g': 'tidak','ga': 'tidak','gak': 'tidak','gk': 'tidak','kaga': 'tidak','kagak': 'tidak',
    'kg': 'tidak','ngga': 'tidak','Nggak': 'tidak','tdk': 'tidak','tak': 'tidak',
    'lgi': 'lagi','lg': 'lagi','donlod': 'download','pdhl': 'padahal','pdhal': 'padahal',
    'Coba2': 'coba-coba','tpi': 'tapi','tp': 'tapi','betmanfaat': 'bermanfaat',
    'gliran': 'giliran','kl': 'kalau','klo': 'kalau','gatau': 'tidak tau','bgt': 'banget',
    'hrs': 'harus','dll': 'dan lain-lain','dsb': 'dan sebagainya','trs': 'terus','trus': 'terus',
    'sangan': 'sangat','bs': 'bisa','bsa': 'bisa','gabisa': 'tidak bisa','gbsa': 'tidak bisa',
    'gada': 'tidak ada','gaada': 'tidak ada','gausah': 'tidak usah','bkn': 'bukan',
    'udh': 'sudah','udah': 'sudah','sdh': 'sudah','pertngahn': 'pertengahan',
    'ribet': 'ruwet','ribed': 'ruwet','sdangkan': 'sedangkan','lemot': 'lambat',
    'lag': 'lambat','ngelag': 'gangguan','yg': 'yang','dipakek': 'di pakai','pake': 'pakai',
    'kya': 'seperti','kyk': 'seperti','ngurus': 'mengurus','jls': 'jelas',
    'burik': 'buruk','payah':'buruk','krna': 'karena','dr': 'dari','smpe': 'sampai',
    'slalu': 'selalu','mulu': 'melulu','d': 'di','konek': 'terhubung','suruh': 'disuruh',
    'apk': 'aplikasi','app': 'aplikasi','apps': 'aplikasi','apl': 'aplikasi',
    'bapuk': 'jelek','bukak': 'buka','nyolong': 'mencuri','pas': 'ketika',
    'uodate': 'update','ato': 'atau','onlen': 'online','cmn': 'cuman','jele': 'jelek',
    'angel': 'susah','jg': 'juga','knp': 'kenapa','hbis': 'setelah','tololl': 'tolol','ny': 'nya',
    'skck':'skck','stnk':'stnk','sim':'sim','sp2hp':'sp2hp','propam':'propam','dumas':'dumas',
    'tilang':'tilang','e-tilang':'tilang','etilang':'tilang','surat kehilangan':'kehilangan'
}

# =============================================================================
# 3. Preprocessing Functions
# =============================================================================
def normalize_repeated_characters(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1", text)

def preprocess_text(text: str) -> str:
    text = str(text)
    text = normalize_repeated_characters(text)
    text = emoji.demojize(text)
    text = re.sub(r":[a-z_]+:", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\@\w+|#", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    text = text.lower()
    for slang, standard in normalization_dict.items():
        text = re.sub(rf"\b{re.escape(slang.lower())}\b", standard.lower(), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text_lda(text: str) -> str:
    # stemming
    text = stemmer.stem(text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords + tokens pendek
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# =============================================================================
# 4. Load Models & Artifacts
# =============================================================================
@st.cache_resource
def load_models():
    # --- LDA ---
    # Assuming lda_model.gensim matches the 4-topic model saved previously
    lda_model = gensim.models.LdaMulticore.load("lda_model.gensim")
    dictionary = corpora.Dictionary.load("lda_dictionary.gensim")

    # --- IndoBERT Sentiment ---
   HF_MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"
try:
    indobert_sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        framework="pt",     # paksa PyTorch
        device=-1           # CPU
    )
except Exception as e:
    indobert_sentiment_pipeline = None
    st.warning(f"Gagal load IndoBERT dari Hugging Face. IndoBERT dimatikan. Detail: {e}")

    # --- LSTM Topic ---
    lstm_topic_model = load_model("lstm_topic_model.h5")
    with open('tokenizer_topic.pkl', 'rb') as f:
        tokenizer_topic = pickle.load(f)
    with open('label_encoder_topic.pkl', 'rb') as f:
        label_encoder_topic = pickle.load(f)

    # --- LSTM Sentiment ---
    lstm_sentiment_model = load_model("lstm_sentiment_model.h5")
    with open('tokenizer_sentiment.pkl', 'rb') as f:
        tokenizer_sentiment = pickle.load(f)
    with open('label_encoder_sentiment.pkl', 'rb') as f:
        label_encoder_sentiment = pickle.load(f)

    return (
        lda_model, dictionary, indobert_pipeline,
        lstm_topic_model, tokenizer_topic, label_encoder_topic,
        lstm_sentiment_model, tokenizer_sentiment, label_encoder_sentiment
    )

(lda_model, dictionary, indobert_pipeline,
 lstm_topic_model, tokenizer_topic, label_encoder_topic,
 lstm_sentiment_model, tokenizer_sentiment, label_encoder_sentiment) = load_models()

# Mappings
topic_name_map = {
    0: "Kemudahan Pengurusan SKCK & Manfaat Aplikasi Polri",
    1: "Efisiensi Pendaftaran Online & Bantuan",
    2: "Isu Teknis, Error & Kendala Penggunaan Aplikasi",
    3: "Kepuasan Layanan Aplikasi & Akses Cepat"
}

# Maxlen for LSTM (must match training)
MAXLEN = 20

# =============================================================================
# 5. Prediction Logic
# =============================================================================

def analyze_review(text):
    results = {}

    # 1. Preprocessing
    clean_text = preprocess_text(text)
    lda_text = preprocess_text_lda(clean_text)

    results['cleaned_text'] = clean_text
    results['lda_text'] = lda_text

    if not lda_text.strip():
        return None

    # 2. LDA Prediction
    bow = dictionary.doc2bow(lda_text.split())
    topics = lda_model.get_document_topics(bow)
    dominant_topic_id = max(topics, key=lambda x: x[1])[0]
    results['lda_topic'] = topic_name_map.get(dominant_topic_id, "Unknown")

    # 3. LSTM Topic Prediction
    seq_topic = tokenizer_topic.texts_to_sequences([clean_text]) # Use cleaned_text as per training
    pad_topic = pad_sequences(seq_topic, maxlen=MAXLEN, padding='post', truncating='post')
    pred_topic = lstm_topic_model.predict(pad_topic)
    topic_idx = np.argmax(pred_topic)
    results['lstm_topic'] = label_encoder_topic.inverse_transform([topic_idx])[0]

    # 4. IndoBERT Sentiment
    sent_bert = indobert_pipeline(clean_text)[0]
    label_map = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    results['indobert_sentiment'] = label_map.get(sent_bert['label'], sent_bert['label'])
    results['indobert_score'] = sent_bert['score']

    # 5. LSTM Sentiment Prediction
    seq_sent = tokenizer_sentiment.texts_to_sequences([clean_text]) # Use cleaned_text
    pad_sent = pad_sequences(seq_sent, maxlen=MAXLEN, padding='post', truncating='post')
    pred_sent = lstm_sentiment_model.predict(pad_sent)
    sent_idx = np.argmax(pred_sent)
    results['lstm_sentiment'] = label_encoder_sentiment.inverse_transform([sent_idx])[0]

    return results

# =============================================================================
# 6. Streamlit UI
# =============================================================================
st.title("🔍 Analisis Ulasan Aplikasi Polri Presisi")
st.markdown("""
Demo ini menggunakan hasil training model:
- **LDA** (Topic Modeling)
- **LSTM** (Topic & Sentiment Classification)
- **IndoBERT** (Sentiment Analysis)
""")

input_text = st.text_area("Masukkan Ulasan Pengguna:", height=150)

if st.button("Analisis"):
    if input_text:
        with st.spinner('Sedang memproses...'):
            res = analyze_review(input_text)
        
        if res:
            st.success("Selesai!")
            
            # Display Cleaned Data
            with st.expander("Lihat Hasil Preprocessing"):
                st.write("**Original:**", input_text)
                st.write("**Cleaned:**", res['cleaned_text'])
                st.write("**LDA Input (Stemmed):**", res['lda_text'])

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📌 Analisis Topik")
                st.info(f"**LDA:** {res['lda_topic']}")
                st.info(f"**LSTM:** {res['lstm_topic']}")
            
            with col2:
                st.subheader("😊 Analisis Sentimen")
                st.success(f"**IndoBERT:** {res['indobert_sentiment']} (Conf: {res['indobert_score']:.4f})")
                st.success(f"**LSTM:** {res['lstm_sentiment']}")
        else:
            st.warning("Teks tidak valid atau kosong setelah preprocessing.")
    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
