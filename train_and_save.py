import re, json, joblib
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

NPM = "20221310099"

stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())

NPM20221310099_NORMALIZE_DICT = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak",
    "yg": "yang", "dgn": "dengan", "aja": "saja", "dr": "dari",
    "krn": "karena", "sm": "sama", "bgt": "banget", "bkn": "bukan"
}

def NPM20221310099_preprocess(text: str, norm_dict: dict) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [norm_dict.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def main():
    print("Loading dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")
    df = pd.DataFrame(dataset["train"])
    df = df.rename(columns={"text": "text", "label": "label"})


    print("Preprocessing...")
    df["text_clean"] = df["text"].apply(lambda x: NPM20221310099_preprocess(x, NPM20221310099_NORMALIZE_DICT))

    X = df["text_clean"].values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    print("Saving artifacts...")
    joblib.dump(model, f"NPM{NPM}_logreg_model.joblib")
    joblib.dump(vectorizer, f"NPM{NPM}_tfidf_vectorizer.joblib")

    with open(f"NPM{NPM}_norm_dict.json", "w", encoding="utf-8") as f:
        json.dump(NPM20221310099_NORMALIZE_DICT, f, ensure_ascii=False, indent=2)

    print("DONE! Files created:")
    print(f"- NPM{NPM}_logreg_model.joblib")
    print(f"- NPM{NPM}_tfidf_vectorizer.joblib")
    print(f"- NPM{NPM}_norm_dict.json")

if __name__ == "__main__":
    main()
