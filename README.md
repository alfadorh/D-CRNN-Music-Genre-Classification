# Dilated-Convolutional Recurrent Neural Network (D-CRNN) for Music Genre Classification

This repository contains the official implementation and documentation of the research paper titled **"Dilated-Convolutional Recurrent Neural Network untuk Klasifikasi Genre Musik"** published in **Jurnal Teknik Informatika dan Sistem Informasi (JuTISI)**, Vol. 10, No. 3, December 2024.

This study evaluates the use of the **Dilated-Convolutional Recurrent Neural Network (D-CRNN)** architecture to automatically classify music genres. This method combines the advantages of **Dilated-CNN** in capturing longer temporal contexts with the temporal sequence recognition capabilities of **CRNN**. 

Utilizing the **GTZAN** dataset, **Mel-Frequency Cepstral Coefficients (MFCC)** feature extraction, and a **sliding-window** data augmentation mechanism, the best proposed model (**MobileNetV2 + D-CRNN**) successfully achieved an accuracy of **89%**, outperforming various music genre classification methods from prior research.

---

## 👥 Authors & Affiliations

* **Mochammad Rizqul Fatichin** (Corresponding Author: 6026231032@student.its.ac.id)
* **Alfado Rafly Hermawan** (6026231041@student.its.ac.id)
* **Raynaldi Anggiat Samuel Siahaan** (5026201037@student.its.ac.id)
* **Rarasmaya Indraswari** (raras@its.ac.id)

**Department of Information Systems, Institut Teknologi Sepuluh Nopember (ITS)** Sukolilo, Surabaya, 60111, Indonesia.

---

## 📌 Abstract

### Bahasa Indonesia
Dalam era digital, pemanfaatan teknologi untuk mengelompokkan genre musik secara otomatis menjadi sangat penting, terutama untuk aplikasi seperti rekomendasi musik, analisis tren musik, dan pengelolaan perpustakaan musik digital. Penelitian ini mengevaluasi penggunaan *Dilated-Convolutional Recurrent Neural Network* (D-CRNN) dalam mengklasifikasi genre musik. Metode ini menggabungkan keunggulan Dilated-CNN dalam menangkap konteks temporal yang lebih panjang dengan kemampuan pengenalan urutan temporal dari CRNN. Data yang digunakan adalah dataset GTZAN yang terdiri dari 1.000 rekaman audio berdurasi 30 detik, dikategorikan ke dalam 10 genre musik. Proses data preprocessing melibatkan konversi rekaman audio menjadi gambar *Mel-Frequency Cepstral Coefficients* (MFCC). Model diuji menggunakan data tanpa augmentasi dan dengan augmentasi, menghasilkan total 15.991 gambar untuk pelatihan. Hasil penelitian menunjukkan bahwa penggunaan D-CRNN dapat meningkatkan akurasi klasifikasi genre musik dibandingkan dengan metode CRNN konvensional.

### English
*In the digital era, utilizing technology to automatically classify music genres has become very important, especially for applications such as music recommendation, music trend analysis, and digital music library management. This research evaluates the use of Dilated-Convolutional Recurrent Neural Network (D-CRNN) in classifying music genres. This method combines the advantages of Dilated-CNN in capturing longer temporal context with the temporal sequence recognition capability of CRNN. The data used is the GTZAN dataset consisting of 1,000 30-second audio recordings, categorized into 10 music genres. Data preprocessing involved converting the audio recordings into Mel-Frequency Cepstral Coefficients (MFCC) images. The model was tested using data without augmentation and with augmentation, resulting in a total of 15.991 images for training. The results show that the use of D-CRNN can improve the accuracy of music genre classification compared to the conventional CRNN method.*

---

## 🚀 Research Workflow (Methodology)

The research workflow consists of three main stages:
1.  **Data Preprocessing**: Converting raw audio into MFCC visual representations.
2.  **Audio Genre Classification**: Training deep learning models using the core D-CRNN architecture alongside transfer learning approaches (MobileNetV2 + D-CRNN and Xception + D-CRNN).
3.  **Results Analysis**: Evaluating metrics such as accuracy, precision, recall, and F1-Score, and comparing them with previous baseline studies.

```
┌──────────────────┐      ┌────────────────────┐      ┌────────────────────────────┐      ┌──────────────────┐
│   GTZAN Audio    │ ───> │ Data Preprocessing │ ───> │ Audio Genre Classification │ ───> │ Results Analysis │
│     Dataset      │      │       (MFCC)       │      │          (D-CRNN)          │      │   & Evaluation   │
└──────────────────┘      └────────────────────┘      └────────────────────────────┘      └──────────────────┘
```

---

## 📊 Dataset & Preprocessing

### 1. GTZAN Dataset
The dataset used is the **GTZAN - Genre Classification Dataset**, sourced from Kaggle.
* **Total Audio**: 1,000 audio files, each 30 seconds long.
* **Format**: `.wav` (Sampling rate: 22,050 Hz, mono channel).
* **Number of Genres**: 10 genres (100 tracks each): *blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock*.

### 2. Preprocessing & Data Augmentation (Sliding-Window)
Audio files are converted into **Mel-Frequency Cepstral Coefficients (MFCC)** visual spectrogram images. Experiments were conducted under two data scenarios:
* **Without Augmentation**: The 30-second audio files were directly converted into MFCC images. Total: **1,000 images**.
* **With Augmentation (Sliding-Window)**: The audio tracks were sliced into 15-second segments using a 1-second interval shift (*stride*). From a single 30-second audio file, **16 segments of 15-second audio** were generated. Total: **15,991 MFCC images**.

### 3. Normalization & Data Split
* **Normalization**: The pixel color scales of the MFCC images (0-255 RGB) were normalized to the range **[0, 1]**.
* **Input Dimensions**: Images were resized to fit the respective architecture input dimensions: **(224, 224, 3)** and **(385, 1162, 3)**.
* **Data Ratio**: Partitioned into Training and Testing sets using an **80:20** split ratio.
    * *Without Augmentation*: Train = 800 images, Test = 200 images.
    * *With Augmentation*: Train = 12,793 images, Test = 3,198 images.

---

## 🧠 Model Architecture

This study evaluates several model architectures:

### 1. Baseline D-CRNN Model
Combines **Dilated Convolutional 2D (DCNN)** layers with **LSTM** and a **Dense Layer (Softmax)**. Dilation in convolutions efficiently expands the *receptive field* without increasing parameters, thereby reducing the risk of overfitting.
* *Optimal Configuration*: 3 Convolutional Layers with filters `(16-32-32)`, a *dilation rate* of `(2-2-2)`, `128` LSTM units, and `128` Dense units.

### 2. MobileNetV2 + D-CRNN (Transfer Learning)
The MobileNetV2 model serves as the feature extractor for the MFCC images. The feature map extraction is truncated at the `block_13_expand_relu` layer, which is then connected to the D-CRNN architecture.

### 3. Xception + D-CRNN (Transfer Learning)
The Xception model serves as the feature extractor for the MFCC images. The feature map extraction is truncated at the `add_10` layer, which is then connected to the D-CRNN architecture.

### Fixed Research Variables (Hyperparameters)
| Parameter | Value |
| :--- | :--- |
| **Train:Test Split** | 80:20 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 |
| **Epochs** | 30 |
| **Optimizer** | Adam |

---

## 📈 Research Results & Evaluation

### 1. Model Performance Without Data Augmentation (Overfitting)
Models trained without augmentation experienced severe *overfitting* issues (training accuracy approached 100%, but validation accuracy remained low):
* **Baseline CRNN**: Validation Accuracy **27.50%**
* **Baseline D-CRNN**: Validation Accuracy **35.00%** (Using 3 convolution layers with 16-32-32 filters and a 2-2-2 dilation rate)
* **MobileNetV2 + D-CRNN**: Validation Accuracy **44.50%** (F1-Score 44%)
* **Xception + D-CRNN**: Validation Accuracy **54.00%** (F1-Score 56%)

### 2. Model Performance With Data Augmentation (Sliding-Window)
After training with the augmented dataset, the abundant data variation successfully mitigated overfitting, leading to a significant drop in validation loss.

| Model | Accuracy | Precision | Recall | F1-Score (Macro Avg) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline D-CRNN Model** | 77% | 79% | 79% | 77% |
| **Xception + D-CRNN** | 80% | 84% | 80% | 80% |
| **MobileNetV2 + D-CRNN** | **89%** | **89%** | **89%** | **89%** |

*Note: All three models demonstrated exceptional classification performance across almost all genres, except for the **rock** genre, which was suboptimally identified by the Baseline D-CRNN and Xception + D-CRNN models.*

### 3. Comparison with Previous Studies
The proposed model successfully outperformed several music genre classification models trained on the GTZAN dataset from prior literature:

| Algorithm | Reference | Input Feature | Accuracy |
| :--- | :--- | :--- | :---: |
| **KNN** | Jakubec & Chmulik (2019) | MFCC | 69% |
| **CNN** | Dong (2019) | Spectrogram | 70% |
| **CRNN** | Adiyansjah et al. (2019) | Mel-Spectrogram | 74% |
| **GLR-CRNN** | Ashraf et al. (2020) | Mel-Spectrogram | 87% |
| **D-CRNN (Proposed)** | Fatichin et al. (2024) | MFCC | **77%** |
| **Xception + D-CRNN (Proposed)** | Fatichin et al. (2024) | MFCC | **80%** |
| **MobileNetV2 + D-CRNN (Proposed)** | Fatichin et al. (2024) | MFCC | **89%** |

---

## 📝 Citation

If you use this method or these findings in your academic work, please cite it as follows:

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