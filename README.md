# Sentiment Analysis (TF-IDF + Logistic Regression) — Debat Capres 2024
**Deployment Streamlit | NPM: 20221310099 | Metode: TF-IDF + Logistic Regression**

Project ini membangun **model analisis sentimen 3 kelas** (NEGATIF/NETRAL/POSITIF) menggunakan pipeline:
**Preprocessing → TF-IDF → Logistic Regression → Prediksi + Probabilitas**, lalu dideploy sebagai aplikasi **Streamlit**.

> Judul use-case mengarah ke “Debat Capres 2024”, tetapi model bisa dipakai untuk **banyak jenis teks** (komentar, review, opini, berita, dsb). Akurasi terbaik diperoleh bila dataset training sejenis dengan data target (bahasa & domain).

---

## Struktur Folder
venv/ # virtual environment (opsional, jangan di-push)
pp.py # aplikasi Streamlit (deployment)
rain_and_save.py # training + generate artifacts
requirements.txt # daftar dependency
NPM20221310099_logreg_model.joblib # artifact: model Logistic Regression
PM20221310099_tfidf_vectorizer.joblib # artifact: TF-IDF vectorizer
NPM20221310099_norm_dict.json # artifact: kamus normalisasi slang→baku
---
## Prasyarat
- Python 3.10+ (disarankan)
- Git (opsional)
- Koneksi internet untuk **training** (download dataset)

---

## 3) Instalasi (Clone / Masuk Folder)
Jika dari GitHub:
```
git clone <repo-url>
cd <your_directory>
```

Jika sudah punya folder lokal:
```
cd "D:\your_directory"
```

## PowerShell (Windows)

Buat venv
```
python -m venv .venv
```

Aktifkan venv
```
.\.venv\Scripts\Activate.ps1
```

Jika muncul error running scripts is disabled:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
.\.venv\Scripts\Activate.ps1
```

Cek venv aktif
```
python --version
pip --version
```


cara Matikan venv
```
deactivate
```

## Git Bash (Windows / MINGW64)

Buat venv
``` bash
python -m venv .venv
```

Aktifkan venv
``` bash
source .venv/Scripts/activate
```
Cek venv aktif
``` bash
python --version
pip --version
```
Cara Matikan venv
``` bas
deactivate
```

Jika di Git Bash python: command not found, gunakan PowerShell atau pastikan Python sudah ter-install dan masuk PATH.


## Install Dependency

Setelah venv aktif:
```
pip install -r requirements.txt
```
Jika requirements.txt belum lengkap, minimal:
```
pip install streamlit scikit-learn joblib sastrawi datasets pandas numpy
```
## Training Model (Generate Artifacts)

Jalankan script training untuk membuat 3 file artifacts .joblib dan .json:
```
python train_and_save.py
```
Jika berhasil, output akan menampilkan:

Validation Accuracy: ...
Saving artifacts...
DONE! Files created: ...

Cek file artifacts sudah terbuat dan tidak 0 bytes

PowerShell
```
dir NPM20221310099_*
```
Git Bash
``` bash
ls -lh NPM20221310099_*
```
Pastikan file berikut ada dan ukurannya bukan 0 bytes:
NPM20221310099_logreg_model.joblib
NPM20221310099_tfidf_vectorizer.joblib
NPM20221310099_norm_dict.json

## Menjalankan Aplikasi Streamlit

Setelah artifacts ada, jalankan:
```
streamlit run app.py
```
Jika ingin membersihkan cache Streamlit:
```
streamlit cache clear
streamlit run app.py
```
Aplikasi akan terbuka di browser (biasanya http://localhost:8501).

## Cara Pakai Aplikasi

Paste teks artikel/komentar pada textarea Input Teks
Klik tombol Prediksi Sentimen
Sistem menampilkan:
Label sentimen (NEGATIF/NETRAL/POSITIF)
Probabilitas tiap kelas
Hasil preprocessing (clean text) pada expander

## Penjelasan Singkat Workflow (Sesuai Kode)
A) Preprocessing (harus konsisten training & deployment)
- Case folding (lowercase)
- Cleaning (hapus URL, mention, hashtag, simbol)
- Tokenization (split kata)
- Normalization (slang→baku via *_norm_dict.json)
- Stopword removal (Sastrawi)
- Stemming (Sastrawi)
- Output preprocessing berupa teks bersih siap TF-IDF.

B) TF-IDF
- Mengubah teks bersih menjadi vektor numerik fitur kata.
Konfigurasi umum:
- max_features=5000
- ngram_range=(1,2) (unigram + bigram)

C) Logistic Regression
- Klasifikasi multiclass (3 label).
- Solver yang digunakan: lbfgs.

## Dataset: Untuk Debat Saja atau Bisa Untuk Banyak Hal?
- Bisa untuk banyak hal: model ini adalah classifier sentimen umum (NEG/NET/POS).
- Debat Capres 2024 adalah contoh use-case (teks artikel debat diprediksi oleh model).
- Akurasi bisa turun jika dataset training berbeda jauh dari data target (perbedaan bahasa/domain/jenis teks).
- Untuk hasil paling baik, latih ulang dengan dataset sentimen bahasa Indonesia dan konteks serupa.
