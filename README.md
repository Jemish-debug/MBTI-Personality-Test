# MBTI Personality Prediction using Machine Learning

## 🧠 Overview

This project presents a machine learning-based approach to predict Myers-Briggs Type Indicator (MBTI) personality types from textual data. Utilizing a Support Vector Machine (SVM) classifier with a linear kernel and TF-IDF feature extraction, the model classifies user-generated text into one of the 16 MBTI personality types. The application is deployed as a Flask web app, providing users with an interactive platform to assess personality types based on their text inputs.





## 📊 Dataset

- **Source**: The dataset comprises 8,675 entries, each containing a user's MBTI type and their corresponding textual posts.
- **Features**:
  - `type`: MBTI personality type (e.g., INFP, ESTJ).
  - `posts`: Aggregated textual posts from users.

## 🛠️ Methodology

### 1. Data Preprocessing

- **Text Cleaning**:
  - Lowercasing text.
  - Removing URLs, special characters, and numbers.
  - Eliminating extra spaces.
  - Removing stopwords using NLTK.
  - Lemmatization to reduce words to their base forms.

### 2. Feature Extraction

- **TF-IDF Vectorization**: Converts textual data into numerical features based on term frequency-inverse document frequency.
- **Additional Features**: Incorporates text length as an additional feature.

### 3. Model Training

- **Algorithm**: Support Vector Machine (SVM) with a linear kernel.
- **Training**: The model is trained on the preprocessed TF-IDF features.
- **Evaluation**:
  - Classification report (precision, recall, F1-score).
  - Confusion matrix.
  - Feature importance analysis.

### 4. Web Application

- **Framework**: Flask.
- **Functionality**:
  - User inputs text data.
  - The model predicts the MBTI personality type.
  - Displays evaluation metrics and visualizations.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

