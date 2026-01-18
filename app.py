import re
import json
import joblib
import numpy as np
import streamlit as st

from io import BytesIO
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================
# IDENTITAS
# =========================
NPM = "20221310099"
PREFIX = f"NPM{NPM}"

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILENAME = f"{PREFIX}_logreg_model.joblib"
VEC_FILENAME   = f"{PREFIX}_tfidf_vectorizer.joblib"
DICT_FILENAME  = f"{PREFIX}_norm_dict.json"

stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())

label_map = {0: "NEGATIF", 1: "NETRAL", 2: "POSITIF"}

# =========================
# PREPROCESSING (Workflow)
# =========================
def NPM20221310099_case_folding(text: str) -> str:
    return str(text).lower()

def NPM20221310099_cleaning(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def NPM20221310099_tokenization(text: str) -> list:
    return text.split()

def NPM20221310099_normalization(tokens: list, norm_dict: dict) -> list:
    return [norm_dict.get(t, t) for t in tokens]

def NPM20221310099_remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def NPM20221310099_lemmatization_stemming(tokens: list) -> list:
    return [stemmer.stem(t) for t in tokens]

def NPM20221310099_preprocess(text: str, norm_dict: dict) -> str:
    text = NPM20221310099_case_folding(text)
    text = NPM20221310099_cleaning(text)
    tokens = NPM20221310099_tokenization(text)
    tokens = NPM20221310099_normalization(tokens, norm_dict)
    tokens = NPM20221310099_remove_stopwords(tokens)
    tokens = NPM20221310099_lemmatization_stemming(tokens)
    return " ".join(tokens)

# =========================
# LOAD ARTIFACTS (Disk / Upload)
# =========================
def NPM20221310099_load_from_disk():
    model_path = BASE_DIR / MODEL_FILENAME
    vec_path   = BASE_DIR / VEC_FILENAME
    dict_path  = BASE_DIR / DICT_FILENAME

    # Validasi existence + size (anti FileNotFound/EOFError)
    for p in [model_path, vec_path, dict_path]:
        if not p.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {p}")
        if p.stat().st_size < 100:
            raise ValueError(f"File terlalu kecil/korup: {p} ({p.stat().st_size} bytes)")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    norm_dict = json.loads(dict_path.read_text(encoding="utf-8"))
    return model, vectorizer, norm_dict

@st.cache_resource
def NPM20221310099_load_artifacts_cached(model_bytes=None, vec_bytes=None, dict_bytes=None):
    """
    Cache agar tidak load ulang terus.
    Bisa load dari disk atau dari upload (bytes).
    """
    if model_bytes is None:
        # dari disk
        return NPM20221310099_load_from_disk()

    # dari upload
    model = joblib.load(BytesIO(model_bytes))
    vectorizer = joblib.load(BytesIO(vec_bytes))
    norm_dict = json.loads(dict_bytes.decode("utf-8"))
    return model, vectorizer, norm_dict

def NPM20221310099_get_artifacts():
    # coba disk
    try:
        model, vectorizer, norm_dict = NPM20221310099_load_artifacts_cached()
        return model, vectorizer, norm_dict, "Loaded from disk ✅"
    except Exception as e:
        # kalau gagal, minta upload
        st.warning("Artifacts belum valid di folder app.py atau file corrupt. Upload ulang 3 file di bawah.")
        st.caption(f"Detail error: {e}")

        up_model = st.file_uploader("Upload model (.joblib)", type=["joblib"])
        up_vec   = st.file_uploader("Upload vectorizer (.joblib)", type=["joblib"])
        up_dict  = st.file_uploader("Upload norm dict (.json)", type=["json"])

        if up_model and up_vec and up_dict:
            model_bytes = up_model.read()
            vec_bytes = up_vec.read()
            dict_bytes = up_dict.read()

            # reset cache bila sebelumnya error
            st.cache_resource.clear()

            model, vectorizer, norm_dict = NPM20221310099_load_artifacts_cached(
                model_bytes=model_bytes,
                vec_bytes=vec_bytes,
                dict_bytes=dict_bytes
            )
            return model, vectorizer, norm_dict, "Loaded from upload ✅"

        st.stop()

# =========================
# PREDICT
# =========================
def NPM20221310099_predict_sentiment(raw_text: str):
    model, vectorizer, norm_dict, status = NPM20221310099_get_artifacts()
    clean_text = NPM20221310099_preprocess(raw_text, norm_dict)
    vec = vectorizer.transform([clean_text])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    return pred, proba, clean_text, status

# =========================
# UI
# =========================
st.set_page_config(page_title=f"Sentiment Analysis - NPM {NPM}", layout="wide")
st.title("Sentiment Analysis (Logistic Regression) — Debat Capres 2024")
st.caption(f"Deployment Streamlit | NPM: {NPM} | Metode: TF-IDF + Logistic Regression")

with st.expander("Info Artifacts (nama file default)"):
    st.write("Taruh file ini satu folder dengan app.py:")
    st.code(f"{MODEL_FILENAME}\n{VEC_FILENAME}\n{DICT_FILENAME}")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Input Teks")
    text_input = st.text_area("Teks", height=300, placeholder="Paste teks artikel/komentar di sini...")
    btn = st.button("Prediksi Sentimen", type="primary", use_container_width=True)

with col2:
    st.subheader("Panduan Cepat")
    st.write("- Masukkan teks artikel/komentar.")
    st.write("- Klik tombol prediksi.")
    st.write("- Sistem menampilkan label + probabilitas.")
    st.divider()
    st.write("Catatan: Model dilatih dari dataset sentiment Indonesia, lalu dipakai memprediksi teks debat.")

if btn:
    if not text_input.strip():
        st.warning("Teks masih kosong.")
    else:
        pred, proba, clean_text, status = NPM20221310099_predict_sentiment(text_input)

        st.info(status)
        st.success(f"Hasil Prediksi: **{label_map.get(pred, str(pred))}**")

        st.write("Probabilitas:")
        st.write({
            "NEGATIF": float(proba[0]),
            "NETRAL": float(proba[1]),
            "POSITIF": float(proba[2]),
        })

        st.progress(float(np.max(proba)))

        with st.expander("Lihat hasil preprocessing (clean text)"):
            st.write(clean_text)
