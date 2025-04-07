import pandas as pd
import joblib
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # Load pre-trained artifacts
    model_data = joblib.load('models/mbti_svm_model.pkl')
    model = model_data['svm_model']
    tfidf = model_data['tfidf_vectorizer']
    le = model_data['label_encoder']
    
    # Load and preprocess data
    df = pd.read_csv("data/train.csv")
    df['cleaned_posts'] = df['posts'].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', '', x.replace("|||", " ").lower()[:500])
    )
    
    # Split data
    _, X_test, _, y_test = train_test_split(
        df['cleaned_posts'], df['type'],
        test_size=0.2, stratify=df['type'], random_state=42
    )
    
    # Transform features
    X_tfidf = tfidf.transform(X_test)
    text_lengths = X_test.apply(len).values.reshape(-1, 1)
    X_final = hstack([X_tfidf, text_lengths])
    
    # Predict
    y_pred = model.predict(X_final)
    y_test_encoded = le.transform(y_test)
    
    # Generate reports
    report = classification_report(
        y_test_encoded, y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/images/confusion_matrix.png')
    plt.close()
    
    # Class metrics plot
    plot_class_metrics(report, le)
    
    # Feature importance plot
    feature_img = plot_feature_importance(model, tfidf, le)
    
    return report, cm, le, feature_img

def plot_class_metrics(report, le):
    classes = le.classes_.tolist()
    metrics = ['precision', 'recall', 'f1-score']
    
    plt.figure(figsize=(15, 6))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        values = [report[cls][metric] for cls in classes]
        plt.barh(classes, values)
        plt.title(f'{metric.capitalize()} per Class')
        plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('static/images/class_metrics.png')
    plt.close()

def plot_feature_importance(model, tfidf, le):
    if not hasattr(model, 'coef_'):
        return None
    
    try:
        coefs = model.coef_
    except AttributeError:
        return None
    
    feature_names = np.append(tfidf.get_feature_names_out(), 'text_length')
    
    plt.figure(figsize=(12, 18))
    for i, cls in enumerate(le.classes_):
        plt.subplot(4, 4, i+1)
        # Convert sparse matrix row to dense array and flatten
        class_coefs = coefs[i].toarray().flatten()  # Fix here
        top_indices = np.argsort(class_coefs)[-10:][::-1]
        top_features = feature_names[top_indices]
        plt.barh(top_features, class_coefs[top_indices])
        plt.title(f'Top Features: {cls}')
    
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png')
    plt.close()
    return 'images/feature_importance.png'

if __name__ == '__main__':
    report, cm, le, _ = evaluate_model()
    print("Classification Report:")
    print(pd.DataFrame(report).transpose().to_markdown())
