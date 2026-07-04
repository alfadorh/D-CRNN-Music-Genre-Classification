# Dilated-Convolutional Recurrent Neural Network (D-CRNN) untuk Klasifikasi Genre Musik

Repository ini berisi implementasi resmi dan dokumentasi dari paper penelitian berjudul **"Dilated-Convolutional Recurrent Neural Network untuk Klasifikasi Genre Musik"** yang diterbitkan pada **Jurnal Teknik Informatika dan Sistem Informasi (JuTISI)**, Vol. 10, No. 3, Desember 2024.

Penelitian ini mengevaluasi penggunaan arsitektur **Dilated-Convolutional Recurrent Neural Network (D-CRNN)** guna mengklasifikasikan genre musik secara otomatis. Metode ini menggabungkan keunggulan **Dilated-CNN** dalam menangkap konteks temporal yang lebih panjang dengan kemampuan pengenalan urutan temporal dari **CRNN**. 

Dengan menggunakan dataset **GTZAN**, ekstraksi fitur **Mel-Frequency Cepstral Coefficients (MFCC)**, dan mekanis augmentasi data **sliding-window**, model terbaik yang diusulkan (**MobileNetV2 + D-CRNN**) berhasil mencapai akurasi sebesar **89%**, mengungguli berbagai metode klasifikasi genre musik pada penelitian-penelitian sebelumnya.

---

## 👥 Penulis & Afiliasi

*   **Mochammad Rizqul Fatichin** (Corresponding Author: 6026231032@student.its.ac.id)
*   **Alfado Rafly Hermawan** (6026231041@student.its.ac.id)
*   **Raynaldi Anggiat Samuel Siahaan** (5026201037@student.its.ac.id)
*   **Rarasmaya Indraswari** (raras@its.ac.id)

**Sistem Informasi, Institut Teknologi Sepuluh Nopember (ITS)**  
Sukolilo, Surabaya, 60111, Indonesia.

---

## 📌 Abstrak / Abstract

### Bahasa Indonesia
Dalam era digital, pemanfaatan teknologi untuk mengelompokkan genre musik secara otomatis menjadi sangat penting, terutama untuk aplikasi seperti rekomendasi musik, analisis tren musik, dan pengelolaan perpustakaan musik digital. Penelitian ini mengevaluasi penggunaan *Dilated-Convolutional Recurrent Neural Network* (D-CRNN) dalam mengklasifikasi genre musik. Metode ini menggabungkan keunggulan Dilated-CNN dalam menangkap konteks temporal yang lebih panjang dengan kemampuan pengenalan urutan temporal dari CRNN. Data yang digunakan adalah dataset GTZAN yang terdiri dari 1.000 rekaman audio berdurasi 30 detik, dikategorikan ke dalam 10 genre musik. Proses data preprocessing melibatkan konversi rekaman audio menjadi gambar *Mel-Frequency Cepstral Coefficients* (MFCC). Model diuji menggunakan data tanpa augmentasi dan dengan augmentasi, menghasilkan total 15.991 gambar untuk pelatihan. Hasil penelitian menunjukkan bahwa penggunaan D-CRNN dapat meningkatkan akurasi klasifikasi genre musik dibandingkan dengan metode CRNN konvensional.

### English
*In the digital era, utilizing technology to automatically classify music genres has become very important, especially for applications such as music recommendation, music trend analysis, and digital music library management. This research evaluates the use of Dilated-Convolutional Recurrent Neural Network (D-CRNN) in classifying music genres. This method combines the advantages of Dilated-CNN in capturing longer temporal context with the temporal sequence recognition capability of CRNN. The data used is the GTZAN dataset consisting of 1,000 30-second audio recordings, categorized into 10 music genres. Data preprocessing involved converting the audio recordings into Mel-Frequency Cepstral Coefficients (MFCC) images. The model was tested using data without augmentation and with augmentation, resulting in a total of 15.991 images for training. The results show that the use of D-CRNN can improve the accuracy of music genre classification compared to the conventional CRNN method.*

---

## 🚀 Alur Penelitian (Methodology)

Alur penelitian terdiri dari tiga tahapan utama:
1.  **Data Preprocessing**: Mengonversi audio mentah menjadi representasi visual MFCC.
2.  **Klasifikasi Genre Audio**: Melatih model deep learning menggunakan arsitektur dasar D-CRNN serta pendekatan transfer learning (MobileNetV2 + D-CRNN dan Xception + D-CRNN).
3.  **Analisis Hasil**: Mengevaluasi metrik akurasi, presisi, recall, dan F1-Score serta membandingkannya dengan penelitian terdahulu.

```
┌──────────────────┐      ┌────────────────────┐      ┌─────────────────────────┐      ┌────────────────┐
│   GTZAN Audio    │ ───> │ Data Preprocessing │ ───> │ Klasifikasi Genre Audio │ ───> │ Analisis Hasil │
│     Dataset      │      │       (MFCC)       │      │        (D-CRNN)         │      │  & Evaluasi    │
└──────────────────┘      └────────────────────┘      └─────────────────────────┘      └────────────────┘
```

---

## 📊 Dataset & Preprocessing

### 1. Dataset GTZAN
Dataset yang digunakan adalah **GTZAN - Genre Classification Dataset** yang diunduh dari Kaggle.
*   **Total Audio**: 1.000 file audio berdurasi 30 detik.
*   **Format**: `.wav` (Sampling rate: 22.050 Hz, mono channel).
*   **Jumlah Genre**: 10 genre (masing-masing 100 trek): *blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, dan rock*.

### 2. Preprocessing & Augmentasi Data (Sliding-Window)
Audio dikonversi menjadi gambar representasi visual **Mel-Frequency Cepstral Coefficients (MFCC)**. Pengujian dilakukan pada dua skenario data:
*   **Tanpa Augmentasi**: File audio 30 detik langsung diubah menjadi gambar MFCC. Total: **1.000 gambar**.
*   **Dengan Augmentasi (Sliding-Window)**: Audio dipotong menjadi segmen berdurasi 15 detik dengan pergeseran (*stride*) interval 1 detik. Dari 1 file audio berdurasi 30 detik, dihasilkan **16 segmen audio berdurasi 15 detik**. Total: **15.991 gambar** MFCC.

### 3. Normalisasi & Split Data
*   **Normalisasi**: Skala warna piksel gambar MFCC (0-255 RGB) dinormalisasi ke rentang **[0, 1]**.
*   **Input Dimensions**: Gambar disesuaikan dengan dimensi input arsitektur, yaitu **(224, 224, 3)** dan **(385, 1162, 3)**.
*   **Rasio Data**: Dibagi menjadi Data Latih dan Data Uji dengan rasio **80:20**.
    *   *Tanpa Augmentasi*: Latih = 800 gambar, Uji = 200 gambar.
    *   *Dengan Augmentasi*: Latih = 12.793 gambar, Uji = 3.198 gambar.

---

## 🧠 Arsitektur Model

Penelitian ini mengevaluasi beberapa arsitektur model:

### 1. Model Dasar D-CRNN
Menggabungkan lapisan **Dilated Convolutional 2D (DCNN)** dengan **LSTM** dan **Dense Layer (Softmax)**. Dilatasi pada konvolusi memperbesar *receptive field* secara efisien tanpa menambah parameter, sehingga mengurangi risiko *overfitting*.
*   *Konfigurasi Terbaik*: 3 Lapisan Konvolusi dengan filter `(16-32-32)`, *dilation rate* `(2-2-2)`, LSTM `128` unit, dan Dense `128` unit.

### 2. MobileNetV2 + D-CRNN (Transfer Learning)
Model MobileNetV2 digunakan sebagai ekstraktor fitur gambar MFCC. Kompleksitas dipotong pada layer `block_13_expand_relu`, kemudian disambung ke arsitektur D-CRNN.

### 3. Xception + D-CRNN (Transfer Learning)
Model Xception digunakan sebagai ekstraktor fitur gambar MFCC. Kompleksitas dipotong pada layer `add_10`, kemudian disambung ke arsitektur D-CRNN.

### Variabel Tetap Penelitian (Hyperparameters)
| Parameter | Nilai |
| :--- | :--- |
| **Train:Test Split** | 80:20 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 |
| **Epoch** | 30 |
| **Optimizer** | Adam |

---

## 📈 Hasil Penelitian & Evaluasi

### 1. Performa Model Tanpa Augmentasi Data (Overfitting)
Model yang dilatih tanpa augmentasi mengalami kendala *overfitting* yang cukup berat (akurasi latih mendekati 100%, namun akurasi validasi rendah):
*   **Baseline CRNN**: Akurasi Validasi **27,50%**
*   **Baseline D-CRNN**: Akurasi Validasi **35,00%** (Menggunakan 3 lapis konvolusi dengan filter 16-32-32 dan dilation rate 2-2-2)
*   **MobileNetV2 + D-CRNN**: Akurasi Validasi **44,50%** (F1-Score 44%)
*   **Xception + D-CRNN**: Akurasi Validasi **54,00%** (F1-Score 56%)

### 2. Performa Model Dengan Augmentasi Data (Sliding-Window)
Setelah dilatih menggunakan data hasil augmentasi, variasi data yang melimpah berhasil mengatasi overfitting secara signifikan dan menurunkan *validation loss*.

| Model | Akurasi | Presisi | Recall | F1-Score (Macro Avg) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Dasar D-CRNN** | 77% | 79% | 79% | 77% |
| **Xception + D-CRNN** | 80% | 84% | 80% | 80% |
| **MobileNetV2 + D-CRNN** | **89%** | **89%** | **89%** | **89%** |

*Catatan: Ketiga model menunjukkan performa klasifikasi yang sangat baik pada hampir seluruh genre, kecuali untuk genre **rock** yang terdeteksi kurang optimal pada Model Dasar D-CRNN dan Xception + D-CRNN.*

### 3. Perbandingan dengan Penelitian Terdahulu
Model yang diusulkan berhasil mengungguli beberapa model klasifikasi genre musik pada dataset GTZAN dari penelitian-penelitian terdahulu:

| Algoritma | Referensi | Fitur Input | Akurasi |
| :--- | :--- | :--- | :---: |
| **KNN** | Jakubec & Chmulik (2019) | MFCC | 69% |
| **CNN** | Dong (2019) | Spectrogram | 70% |
| **CRNN** | Adiyansjah et al. (2019) | Mel-Spectrogram | 74% |
| **GLR-CRNN** | Ashraf et al. (2020) | Mel-Spectrogram | 87% |
| **D-CRNN (Ousulan)** | Fatichin et al. (2024) | MFCC | **77%** |
| **Xception + D-CRNN (Usulan)** | Fatichin et al. (2024) | MFCC | **80%** |
| **MobileNetV2 + D-CRNN (Usulan)** | Fatichin et al. (2024) | MFCC | **89%** |

---

## 📝 Sitasi (Citation)

Jika Anda menggunakan metode atau hasil penelitian ini dalam karya akademis Anda, harap berikan sitasi sebagai berikut:

```bibtex
@article{fatichin2024dcrnn,
  author    = {Mochammad Rizqul Fatichin and Alfado Rafly Hermawan and Raynaldi Anggiat Samuel Siahaan and Rarasmaya Indraswari},
  title     = {Dilated-Convolutional Recurrent Neural Network untuk Klasifikasi Genre Musik},
  journal   = {Jurnal Teknik Informatika dan Sistem Informasi (JuTISI)},
  volume    = {10},
  number    = {3},
  pages     = {439--448},
  year      = {2024},
  doi       = {10.28932/jutisi.v10i3.9347},
  issn      = {e-ISSN: 2443-2229, p-ISSN: 2443-2210}
}
```
