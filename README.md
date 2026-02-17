project:
  title: Deep Learning Approaches for Automated Misinformation Detection
  description: >
    A comparative study of classical machine learning and transformer-based
    deep learning techniques for fake news classification.

overview:
  dataset: Kaggle Fake and Real News Dataset
  models:
    - TF-IDF + Logistic Regression (Baseline)
    - BERT (bert-base-uncased Transformer Model)
  deployment:
    backend: Flask REST API
    frontend: React Native Mobile Application

motivation: >
  The rapid spread of misinformation across digital platforms poses significant
  societal challenges. Automated detection systems leveraging contextual language
  models can assist in mitigating fake news dissemination. This project explores
  both traditional and transformer-based deep learning methods for robust classification.

experimental_results:
  - model: Logistic Regression
    accuracy: 98.78%
  - model: BERT (Transformer)
    accuracy: 99.98%

model_architecture:
  baseline_model:
    text_vectorization: TF-IDF
    classifier: Logistic Regression
  deep_learning_model:
    pretrained_model: bert-base-uncased
    max_sequence_length: 256
    optimizer: AdamW
    loss_function: Cross-Entropy

dataset:
  name: Kaggle Fake and Real News Dataset
  included_in_repo: false
  download_link: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

project_structure:
  - backend: Flask API and inference logic
  - detection: Training scripts and evaluation
  - fake-news-app: React Native mobile frontend

setup:
  backend:
    commands:
      - cd backend
      - pip install -r requirements.txt
      - python app.py
  mobile_app:
    commands:
      - cd fake-news-app
      - npm install
      - expo start

research_contribution:
  - Comparative analysis between classical ML and Transformer models.
  - Demonstration of contextual embedding effectiveness.
  - Deployment-ready inference system with REST API.
  - Mobile interface for real-time news verification.

author:
  name: Surya
  qualification: B.Tech Computer Science and Engineering

license:
  type: Academic and Research Use
  description: This project is intended for academic and research purposes.
