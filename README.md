# Deep Learning Approaches for Automated Misinformation Detection

A comparative study of classical machine learning and transformer-based deep learning techniques for fake news classification.

---

## 📌 Overview

This project implements and evaluates multiple approaches for automated misinformation detection using the Kaggle Fake and Real News Dataset.

The study compares:

- TF-IDF + Logistic Regression (Baseline)
- BERT (`bert-base-uncased` Transformer Model)

The system is deployed using:

- Flask REST API (Backend)
- React Native Mobile Application (Frontend)

---

## 🧠 Motivation

The rapid spread of misinformation across digital platforms poses significant societal challenges. Automated detection systems leveraging contextual language models can assist in mitigating fake news dissemination.

This project explores both traditional and transformer-based deep learning methods for robust classification.

---

## 📊 Experimental Results

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | 98.78% |
| BERT (Transformer) | 99.98% |

The high performance is attributed to contextual semantic modeling and structured dataset characteristics.

---

## 📈 Model Architecture

### Baseline Model
- Text Vectorization: TF-IDF
- Classifier: Logistic Regression

### Deep Learning Model
- Pretrained Model: `bert-base-uncased`
- Fine-tuned for binary classification
- Max Sequence Length: 256
- Optimizer: AdamW
- Loss Function: Cross-Entropy

---

## 📂 Dataset

Kaggle Fake and Real News Dataset.

Due to size constraints, the dataset is not included in this repository.

Download link:  
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

---

## ⚙️ Project Structure

