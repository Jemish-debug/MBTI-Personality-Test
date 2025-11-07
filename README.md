# MBTI Personality Prediction Web Application

This project predicts **Myers–Briggs Type Indicator (MBTI)** personality types based on user-provided text input using **classical machine learning** techniques. The application features a clean web interface built with **Flask**, a **Support Vector Machine (SVM)** model for text classification, and an additional **integrated chatbot** for interactive engagement and fun.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-Flask-red.svg" />
  <img src="https://img.shields.io/badge/Model-SVM-yellow.svg" />
  <img src="https://img.shields.io/badge/Feature%20Extraction-TF--IDF-green.svg" />
  <img src="https://img.shields.io/badge/Chatbot-Gemini%20AI-purple.svg" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" />
</p>


## 🌟 Features

- Predicts one of **16 MBTI personality types** from text input.
- Uses **TF-IDF** + **SVM** for efficient and interpretable text classification.
- Simple and user-friendly **Flask web interface** for real-time predictions.
- **Integrated chatbot** powered by Google Gemini for conversational interaction.
- Includes visual evaluation metrics:
  - Confusion Matrix
  - Class-wise Precision/Recall/F1
  - Feature Importance Plots

## 🧠 How It Works

1. The user enters text describing their thoughts or writing style.
2. The input text is cleaned and converted into TF-IDF vectors.
3. An SVM classifier predicts the MBTI type.
4. The result is displayed along with personality interpretation.
5. Users can also open the **Chatbox** page to interact with the built-in chatbot.

## 🤖 Chatbot Feature

The project includes a **lightweight chatbot** for interactive fun and conversation:
- Accessible via `/chatbox`
- Powered by **Google Generative AI (Gemini)**
- Supports open conversational queries

This chatbot is **separate** from the MBTI model and does not influence predictions.

## 🛠️ Tech Stack

**Python, Flask, scikit-learn, pandas, NumPy, joblib, matplotlib, seaborn, Google Generative AI**

## 📂 Project Structure

```
mbti-personality-prediction/
├── app.py
├── evaluate.py
├── train.csv
├── models/
│   └── mbti_svm_model.pkl
├── templates/
│   ├── index.html
│   └── chatbox.html
├── static/
│   └── images/
│       ├── confusion_matrix.png
│       ├── class_metrics.png
│       └── feature_importance.png
├── requirements.txt
├── README.md
```

## 🚀 Getting Started

### 1) Clone the Repository
```bash
git clone <your-repo-url>
cd mbti-personality-prediction
```

### 2) Install Dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the App
```bash
python app.py
```

### 4) Open in Browser
```
http://localhost:5000
```

## 🎯 Future Enhancements

- UI/UX improvements
- Larger dataset for better prediction accuracy
- Support for multiple languages
- Option for long-form personality analysis

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📝 License

This project is licensed under the **MIT License**.
