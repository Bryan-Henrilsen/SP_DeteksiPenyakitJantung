# 🫀 Sistem Pakar Deteksi Penyakit Jantung

Aplikasi web berbasis Streamlit untuk mendeteksi kemungkinan penyakit jantung berdasarkan data masukan pengguna. Aplikasi ini menggunakan model Machine Learning yang telah dilatih untuk memprediksi risiko penyakit jantung.

---

## 📌 Fitur
- Input data pengguna secara interaktif (usia, tekanan darah, kolesterol, dll)
- Prediksi risiko penyakit jantung berdasarkan model Gaussian Naive Bayes
- Interface sederhana dan mudah digunakan
- Dapat dijalankan secara lokal dengan Streamlit

---

## 🚀 Cara Menjalankan (Local)

1. **Clone repository ini:**

```bash
git clone https://github.com/Bryan-Henrilsen/SP_DeteksiPenyakitJantung.git
cd SP_DeteksiPenyakitJantung

python -m venv venv
# Aktifkan environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Library yang Dibutuhkan
pip install -r requirements.txt

streamlit run jantung_app.py
```
