import joblib as jb
import streamlit as st
import numpy as np
import pandas as pd

# JUDUL APLIKASI
st.set_page_config(page_title="Sistem Pakar Deteksi Penyakit Jantung", page_icon="icon/heart.png", layout="wide")
st.title("Sistem Pakar Deteksi Penyakit Jantung")
st.markdown("""
### 🫀 Tentang Aplikasi Ini

Aplikasi ini menggunakan model Naive Bayes untuk memprediksi adanya penyakit jantung berdasarkan input 13 fitur klinis. .  
Data yang digunakan berdasarkan dataset Heart Disease Statlog dari UCI repository (https://archive.ics.uci.edu/ml/datasets/statlog+(heart)).

Silakan isi data pasien di panel sebelah kiri, dan sistem akan memberikan hasil prediksi serta penjelasan singkatnya.

---
""")

# MEMUAT MODEL YANG TELAH DILATIH
try: 
    model = jb.load('model_jantung.joblib')
except FileNotFoundError:
    st.error("File model 'model_jantung.joblib' tidak ditemukan.")
    st.stop()

# MEMUAT DATASET YANG DIGUNAKAN UNTUK FITUR REKOMENDASI
try:
    dataset = pd.read_csv('Data/Heart_disease_statlog.csv')
    feature_means = dataset.mean(numeric_only=True)
except FileNotFoundError:
    st.warning("⚠️ Dataset tidak ditemukan. Rekomendasi tidak bisa ditampilkan.")
    feature_means = None

# MEMBUAT FORM INPUT DI SIDEBAR
st.sidebar.header("Input Data Pasien")

input_data = {}

input_data['age'] = st.sidebar.number_input(
    "Usia (tahun)", 1, 120, 50, 
    help="Masukkan usia pasien dalam tahun"
)
input_data['sex'] = st.sidebar.selectbox(
    "Jenis Kelamin", 
    options=[(0, "Perempuan"), (1, "Laki-Laki")], 
    format_func=lambda x: x[1]
)[0]
input_data['cp'] = st.sidebar.selectbox(
    "Tipe Nyeri Dada", 
    options=[
        (0, "Angina Tipikal (Terkait aktivitas)"),
        (1, "Angina Atipikal (Tidak khas)"),
        (2, "Nyeri Non-Angina (Tidak terkait jantung)"),
        (3, "Asimtomatik (Tanpa nyeri dada)")
    ],
    format_func=lambda x: x[1],
    help="Jenis nyeri dada yang dialami"
)[0]
input_data['trestbps'] = st.sidebar.slider(
    "Tekanan Darah Istirahat (mm Hg)", 80, 200, 120, 
    help="Tekanan darah saat istirahat"
)
input_data['chol'] = st.sidebar.slider(
    "Kadar Kolesterol Serum (mg/dl)", 100, 600, 200,
    help="Kadar kolesterol dalam darah"
)
input_data['fbs'] = st.sidebar.selectbox(
    "Gula Darah Puasa > 120 mg/dl", 
    options=[(0, "Tidak"), (1, "Ya")], 
    format_func=lambda x: x[1],
    help="Apakah kadar gula darah puasa melebihi 120 mg/dl"
)[0]
input_data['restecg'] = st.sidebar.selectbox(
    "Hasil EKG saat Istirahat", 
    options=[
        (0, "Normal"), 
        (1, "Kelainan ST-T"), 
        (2, "Hipertrofi Ventrikel Kiri")
    ],
    format_func=lambda x: x[1],
    help="Hasil pemeriksaan EKG saat istirahat"
)[0]
input_data['thalach'] = st.sidebar.slider(
    "Detak Jantung Maksimum", 60, 220, 150,
    help="Detak jantung tertinggi saat tes treadmill"
)
input_data['exang'] = st.sidebar.selectbox(
    "Nyeri Dada Saat Olahraga", 
    options=[(0, "Tidak"), (1, "Ya")], 
    format_func=lambda x: x[1],
    help="Apakah pasien mengalami nyeri dada selama aktivitas fisik?"
)[0]
input_data['oldpeak'] = st.sidebar.slider(
    "Oldpeak (Penurunan ST)", 0.0, 6.2, 1.0, step=0.1,
    help="Perbedaan segmen ST saat istirahat dan saat olahraga (indikator aliran darah ke jantung)"
)
input_data['slope'] = st.sidebar.selectbox(
    "Kemiringan ST", 
    options=[
        (0, "Menanjak (Normal)"), 
        (1, "Datar (Perlu diperhatikan)"), 
        (2, "Menurun (Waspada)")
    ], 
    format_func=lambda x: x[1],
    help="Bentuk kemiringan segmen ST pada EKG saat puncak latihan"
)[0]
input_data['ca'] = st.sidebar.selectbox(
    "Jumlah Pembuluh Darah Terlihat", 
    options=[0, 1, 2, 3],
    help="Jumlah pembuluh darah utama yang terlihat saat fluoroskopi (0–3)"
)
input_data['thal'] = st.sidebar.selectbox(
    "Status Thalassemia", 
    options=[
        (1, "Normal (Aliran darah baik)"), 
        (2, "Cacat Permanen (Tidak ada aliran)"), 
        (3, "Cacat Reversibel (Aliran tidak normal)")
    ],
    format_func=lambda x: x[1],
    help="Jenis kelainan aliran darah jantung (Thalassemia)"
)[0]

# MEMBUAT TOMBOL PREDIKSI & MENAMPILKAN HASIL
if st.sidebar.button("Lakukan Diagnosis"):
    features = np.array(list(input_data.values())).reshape(1, -1)
    
    pred_class = model.predict(features)[0]
    pred_probs = model.predict_proba(features)[0]
    
    confidence_score = pred_probs[pred_class]
    
    st.subheader("📊 Hasil Diagnosis")
    
    if pred_class == 0:
        st.success(f"**Diagnosis:** Normal (Tidak Terdeteksi Penyakit Jantung)")
    else:
        st.error(f"**Diagnosis:** Terdeteksi Adanya Penyakit Jantung")
    
    st.write(f"**Tingkat Kepercayaan:** {confidence_score:.2%}")
    
    st.subheader("🔍 Detail Probabilitas")
    st.write(f"Probabilitas Normal: {pred_probs[0]:.2%}")
    st.write(f"Probabilitas Sakit Jantung: {pred_probs[1]:.2%}")
    
    if feature_means is not None:
        st.subheader("💡 Rekomendasi")
        rekomendasi = []
        for key in ['trestbps', 'chol', 'thalach', 'oldpeak']:
            user_val = input_data[key]
            mean_val = feature_means[key]
            
            if key in ['thalach']:  # thalach lebih tinggi lebih baik
                if user_val < mean_val:
                    rekomendasi.append(f"🔴 **{key.upper()}** di bawah rata-rata ({user_val} < {mean_val:.1f}). Perlu peningkatan untuk kebugaran jantung.")
                else:
                    rekomendasi.append(f"🟢 **{key.upper()}** di atas rata-rata ({user_val} ≥ {mean_val:.1f}). Ini baik, pertahankan!")
            else:  # lainnya: makin kecil makin baik
                if user_val > mean_val:
                    rekomendasi.append(f"🔴 **{key.upper()}** di atas rata-rata ({user_val} > {mean_val:.1f}). Perhatikan gaya hidup & konsultasikan dengan dokter.")
                else:
                    rekomendasi.append(f"🟢 **{key.upper()}** di bawah rata-rata ({user_val} ≤ {mean_val:.1f}). Ini baik, pertahankan!")

        for r in rekomendasi:
            st.markdown(r)
    
    # REKOMENDASI FITUR KATEGORIKAL
    st.markdown("### 🔎 Catatan:")
    
    if input_data['sex'] == 1:
        st.markdown("⚠️ **Jenis Kelamin:** Laki-laki memiliki risiko lebih tinggi terhadap penyakit jantung. Waspadai gejala lebih awal.")
    else:
        st.markdown("✅ **Jenis Kelamin:** Perempuan umumnya memiliki risiko sedikit lebih rendah.")

    if input_data['cp'] in [2, 3]:
        st.markdown("⚠️ **Tipe Nyeri Dada:** Jenis nyeri non-angina atau asimtomatik menunjukkan gejala yang perlu perhatian khusus.")
    elif input_data['cp'] == 1:
        st.markdown("🟡 **Tipe Nyeri Dada:** Nyeri atipikal perlu observasi lanjut.")
    else:
        st.markdown("✅ **Tipe Nyeri Dada:** Nyeri khas yang masih bisa ditangani sesuai gejala.")

    if input_data['fbs'] == 1:
        st.markdown("⚠️ **Gula Darah Puasa Tinggi:** Indikasi potensi diabetes atau gangguan metabolik. Jaga pola makan.")
    else:
        st.markdown("✅ **Gula Darah Puasa:** Dalam batas normal.")

    if input_data['restecg'] == 1:
        st.markdown("⚠️ **EKG:** Terdapat kelainan gelombang ST-T. Perlu evaluasi medis lebih lanjut.")
    elif input_data['restecg'] == 2:
        st.markdown("🔴 **EKG:** Hipertrofi ventrikel kiri ditemukan. Ini serius dan harus dikonsultasikan.")
    else:
        st.markdown("✅ **EKG:** Hasil normal.")

    if input_data['exang'] == 1:
        st.markdown("⚠️ **Nyeri Dada Saat Olahraga:** Gejala klasik gangguan jantung. Harap tidak abaikan.")
    else:
        st.markdown("✅ **Aktivitas Fisik:** Tidak menunjukkan gejala saat berolahraga.")

    if input_data['slope'] == 2:
        st.markdown("🔴 **Kemiringan ST Menurun:** Risiko tinggi. Perlu pemeriksaan jantung lanjutan.")
    elif input_data['slope'] == 1:
        st.markdown("🟡 **Kemiringan ST Datar:** Perlu diperhatikan lebih lanjut.")
    else:
        st.markdown("✅ **Kemiringan ST Normal:** Menanjak adalah kondisi normal.")

    if input_data['ca'] >= 2:
        st.markdown(f"✅ **Pembuluh Darah Terlihat: {input_data['ca']}** – Kondisi sirkulasi masih terpantau normal.")
    else:
        st.markdown(f"⚠️ **Pembuluh Darah Terlihat: {input_data['ca']}** – Semakin banyak yang tidak terlihat, semakin tinggi risiko penyumbatan.")

    if input_data['thal'] == 2:
        st.markdown("🔴 **Thalassemia - Cacat Permanen:** Tidak ada aliran darah ke bagian jantung tertentu. Ini sangat serius.")
    elif input_data['thal'] == 3:
        st.markdown("🟡 **Thalassemia - Cacat Reversibel:** Aliran darah tidak normal saat beraktivitas. Perlu perawatan medis.")
    else:
        st.markdown("✅ **Thalassemia:** Aliran darah normal.")

st.sidebar.info("Pastikan semua nilai input sudah sesuai sebelum melakukan diagnosis")
st.info("🧠 Aplikasi ini hanya sebagai alat bantu. Untuk diagnosis pasti, hubungi tenaga medis profesional.")
