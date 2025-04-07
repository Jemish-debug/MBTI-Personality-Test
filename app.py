# app.py
# AIzaSyDAt2ENsSi2y5vWCNHDpHk_7Yb0Y34WZ3Q
from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import re
import numpy as np
from scipy.sparse import hstack
import os
from datetime import datetime
import google.generativeai as genai
import evaluate

app = Flask(__name__)
app.secret_key = os.urandom(24)
GEMINI_API_KEY = "AIzaSyDAt2ENsSi2y5vWCNHDpHk_7Yb0Y34WZ3Q"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the generative model for chat responses
chat_model = genai.GenerativeModel('gemini-1.5-flash')

def load_model():
    model_data = joblib.load('models/mbti_svm_model.pkl')
    return (
        model_data['svm_model'],
        model_data['tfidf_vectorizer'],
        model_data['label_encoder']
    )

# Load classifier model separately and assign to classifier_model
classifier_model, tfidf, le = load_model()

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = text.replace("|||", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()[:500]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predictPersonality():
    prediction = None
    if request.method == 'POST':
        user_text = request.form['text']
        cleaned_text = preprocess_text(user_text)
        
        # Transform text
        tfidf_features = tfidf.transform([cleaned_text])
        text_length = np.array([len(cleaned_text)]).reshape(-1, 1)
        features = hstack([tfidf_features, text_length])
        
        # Predict personality type
        pred_num = classifier_model.predict(features)[0]
        mbti_type = le.inverse_transform([pred_num])[0]
        
        # Get type description
        type_descriptions = {
            'ISTJ': 'The Logistician - Practical, detail-oriented, and dependable. Thrives on order, rules, and responsibility. A rock of stability who honors traditions.',
            'ISFJ': 'The Defender - Warm, loyal, and conscientious. Protects loved ones tirelessly. Excels in caregiving and remembers details about others.',
            'INFJ': 'The Advocate - Idealistic visionaries with deep empathy. Quietly inspire change while valuing integrity. Rare personality type (1 of population).',
            'INTJ': 'The Architect - Strategic mastermind with relentless logic. Solves complex problems through original thinking. Independent and future-focused.',
            'ISTP': 'The Virtuoso - Bold hands-on troubleshooters. Masters tools and thrives in crises. "Living in the moment" mechanics of the world.',
            'ISFP': 'The Adventurer - Gentle artists of action. Values aesthetics and experiences. Quietly rebellious with a strong connection to nature.',
            'INFP': 'The Mediator - Poetic idealists driven by values. Seeks harmony and meaning. Champions for causes through creative expression.',
            'INTP': 'The Logician - Innovative theorist obsessed with analysis. Redefines systems through abstract thinking. Loves intellectual debate.',
            'ESTP': 'The Entrepreneur - Energetic thrill-seekers. Masters of persuasion and quick thinking. Lives by the motto: "Work hard, play harder."',
            'ESFP': 'The Entertainer - Spontaneous life of the party. Radiates enthusiasm and charm. Turns mundane moments into joyful experiences.',
            'ENFP': 'The Champion - People-inspired innovators. Sees potential in everyone. Motivates others through infectious optimism and creativity.',
            'ENTP': 'The Debater - Quick-witted intellectual rebels. Challenges norms through relentless curiosity. "Devil\'s advocate who loves mental sparring."',
            'ESTJ': 'The Executive - Natural-born organizers. Upholds standards through practical leadership. "Get it done" mentality with clear structure.',
            'ESFJ': 'The Consul - Warm-hearted social coordinators. Creates harmony through tradition. Pillars of community who remember birthdays.',
            'ENFJ': 'The Protagonist - Charismatic mentors. Reads emotions effortlessly. Motivates teams toward collective growth and ethical goals.',
            'ENTJ': 'The Commander - Bold strategic leaders. Converts vision into action. Natural CEOs who optimize systems for maximum efficiency.'
        }
        prediction = {
            'type': mbti_type,
            'description': type_descriptions.get(mbti_type, '')
        }
    
    return render_template('predict.html', prediction=prediction)

@app.route('/evaluate')
def evaluation_page():
    report, cm, le_obj, feature_img = evaluate.evaluate_model()
    
    # Convert le.classes_ to list for membership check
    class_labels = le_obj.classes_.tolist()
    
    class_report = [
        {
            'type': k,
            'precision': v['precision'],
            'recall': v['recall'],
            'f1': v['f1-score'],
            'support': v['support']
        } for k, v in report.items() if k in class_labels  # Use converted list
    ]
    
    accuracy = report['accuracy']
    
    return render_template(
        'evaluate.html',
        class_report=class_report,
        accuracy=accuracy,
        confusion_matrix='images/confusion_matrix.png',
        class_metrics='images/class_metrics.png',
        feature_importance=feature_img
    )

@app.route('/aboutus')
def aboutUs():
    return render_template('aboutus.html')

@app.route('/chatbox', methods=['GET', 'POST'])
def chat_interface():
    # Initialize messages in session if they don't exist
    if 'messages' not in session:
        session['messages'] = []
    
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        if user_input:
            try:
                # Use the chat_model to generate a response to the user's input
                response = chat_model.generate_content(user_input)
                ai_response = response.text
                
                # Store messages with timestamp
                timestamp = datetime.now().strftime("%H:%M")
                session['messages'].extend([
                    {'text': user_input, 'is_user': True, 'timestamp': timestamp},
                    {'text': ai_response, 'is_user': False, 'timestamp': timestamp}
                ])
                session.modified = True
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                session['messages'].append({
                    'text': error_msg,
                    'is_user': False,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
                
    return render_template('chatbox.html', messages=session['messages'])

if __name__ == '__main__':
    app.run(debug=True)
