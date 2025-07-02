import joblib as jb
import streamlit as st
import numpy as np
import pandas as pd

# JUDUL APLIKASI
st.set_page_config(page_title="Sistem Pakar Deteksi Penyakit Jantung", page_icon="icon/heart.png", layout="wide")
st.title("Sistem Pakar Deteksi Penyakit Jantung")
st.write("Aplikasi ini menggunakan model Naive Bayes untuk memprediksi adanya penyakit jantung berdasarkan input 13 fitur klinis.")

# MEMUAT MODEL YANG TELAH DILATIH
try: 
    model = jb.load('model_jantung.joblib')
except FileNotFoundError:
    st.error("File model 'model_jantung.joblib' tidak ditemukan.")
    st.stop()

# MEMBUAT FORM INPUT DI SIDEBAR
st.sidebar.header("Input Data Pasien")

# BUAT INPUT UNTUK SETIAP FITUR
input_data = {}

input_data['age'] = st.sidebar.number_input("Usia (tahun)", 1, 120, 50)
input_data['sex'] = st.sidebar.selectbox("Jenis Kelamin", options=[(0, "Perempuan"), (1, "Laki-Laki")], format_func=lambda x: x[1])[0]
input_data['cp'] = st.sidebar.selectbox("Tipe Nyeri Dada", options=[(0, "Angina Tipikal"), (1, "Angina Atipikal"), (2, "Nyeri Non-Angina"), (3, "Asimtomatik")], format_func=lambda x: x[1])[0]
input_data['trestbps'] = st.sidebar.slider("Tekanan Darah Istirahat (mm Hg)", 80, 200, 120)
input_data['chol'] = st.sidebar.slider("Kolestrol Serum (mg/dl)", 100, 600, 200)
input_data['fbs'] = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dl", options=[(0, "Tidak"), (1, "Ya")], format_func=lambda x: x[1])[0]
input_data['restecg'] = st.sidebar.selectbox("Hasil EKG Istirahat", options=[(0, "Normal"), (1, "Abnormalitas ST-T"), (2, "Hipertrofi Ventrikel Kiri")], format_func=lambda x: x[1])[0]
input_data['thalach'] = st.sidebar.slider("Detak Jantung Maksimum", 60, 220, 150)
input_data['exang'] = st.sidebar.selectbox("Angina Akibat Olahraga", options=[(0, "Tidak"), (1, "Ya")], format_func=lambda x: x[1])[0]
input_data['oldpeak'] = st.sidebar.slider("Oldpeak (Depresi ST)", 0.0, 6.2, 1.0, step=0.1)
input_data['slope'] = st.sidebar.selectbox("Kemiringan ST", options=[(0, "Menanjak"), (1, "Datar"), (2, "Menurun")], format_func=lambda x: x[1])[0]
input_data['ca'] = st.sidebar.selectbox("Jumlah Pembuluh Darah Utama", options=[0, 1, 2, 3])
input_data['thal'] = st.sidebar.selectbox("Status Thalassemia", options=[(1, "Normal"), (2, "Cacat Permanen"), (3, "Cacat Reversibel")], format_func=lambda x: x[1])[0]

# MEMBUAT TOMBOL PREDIKSI & MENAMPILKAN HASIL
if st.sidebar.button("Lakukan Prediksi"):
    # MENGUBAH INPUT SETIAP FITUR MENJADI ARRAY NUMPY
    features = np.array(list(input_data.values())).reshape(1, -1)
    
    # LAKUKAN PREDIKSI
    pred_class = model.predict(features)[0]
    pred_probs = model.predict_proba(features)[0]
    
    confidence_score = pred_probs[pred_class]
    
    st.subheader("📊 Hasil Prediksi")
    
    if pred_class == 0:
        st.success(f"**Diagnosis:** Normal (Tidak Terdeteksi Penyakit Jantung)")
    else:
        st.error(f"**Diagnosis:** Terdeteksi Adanya Penyakit Jantung")
    
    st.write(f"**Tingkat Kepercayaan:** {confidence_score:.2%}")
    
    # MENAMPILKAN DETAIL PROBABILITAS
    st.subheader("🔍 Detail Probabilitas")
    st.write(f"Probabilitas Normal: {pred_probs[0]:.2%}")
    st.write(f"Probabilitas Sakit Jantung: {pred_probs[1]:.2%}")
    
st.sidebar.info("Pastikan semua nilai input sudah sesuai sebelum melakukan prediksi")