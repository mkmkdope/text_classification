# app.py
import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (if not already downloaded)
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
except:
    pass

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# IMPORTANT: This must match EXACTLY with the preprocessing used during training
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def predict_toxicity(text, model, vectorizer):
    # Preprocess the text (using the same function as during training)
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_vec = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vec)
    
    # Try to get probabilities if available
    if hasattr(model, 'predict_proba'):
        prediction_proba = model.predict_proba(text_vec)
    else:
        prediction_proba = None
    
    # Format results
    results = {}
    target_labels = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    
    for i, label in enumerate(target_labels):
        result_item = {
            'prediction': bool(prediction[0, i])
        }
        
        if prediction_proba is not None:
            result_item['probability'] = float(prediction_proba[0, i])
        
        results[label] = result_item
    
    # Determine if overall toxic
    toxic = any(prediction[0])
    
    return {
        'is_toxic': toxic,
        'breakdown': results
    }

# Load models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("models/vectorizer.pkl")
        models = {}
        models['Naive Bayes'] = joblib.load("models/naive_bayes_model.pkl")
        models['SVM'] = joblib.load("models/svm_model.pkl")
        models['Logistic Regression'] = joblib.load("models/logistic_regression_model.pkl")
        return vectorizer, models
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}. Please run the training script first.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.set_page_config(page_title="Toxicity Detector", page_icon="⚠️", layout="wide")
    
    st.title("🔍 Toxicity Detection App")
    st.markdown("Analyze text for various types of toxic content using machine learning models")
    
    # Load models
    vectorizer, models = load_models()
    
    if vectorizer is None or models is None:
        st.warning("Could not load models. Please make sure:")
        st.write("1. You've run the training notebook (nb_svm.ipynb)")
        st.write("2. The model files are in the 'models' folder")
        st.write("3. The vectorizer.pkl file exists and is valid")
        return
    
    # Sidebar for model selection
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        list(models.keys())
    )
    
    # Add info about the selected model
    model_info = {
        'Naive Bayes': 'Fast and works well with text data',
        'SVM': 'Good for high-dimensional data like text',
        'Logistic Regression': 'Provides probability estimates'
    }
    st.sidebar.info(f"{selected_model}: {model_info[selected_model]}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter text to analyze")
        text_input = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="Enter text to check for toxicity..."
        )
        
        if st.button("Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    try:
                        result = predict_toxicity(text_input, models[selected_model], vectorizer)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Overall toxicity
                        if result['is_toxic']:
                            st.error("🚨 TOXIC CONTENT DETECTED")
                        else:
                            st.success("✅ No toxicity detected")
                        
                        # Detailed breakdown
                        st.subheader("Detailed Breakdown")
                        
                        for label, data in result['breakdown'].items():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if data['prediction']:
                                    st.error(f"**{label.replace('_', ' ').title()}**: ✅ Detected")
                                else:
                                    st.info(f"**{label.replace('_', ' ').title()}**: ❌ Not detected")
                            
                            with col2:
                                if 'probability' in data:
                                    # Create a visual indicator for probability
                                    prob_value = data['probability']
                                    st.metric("Confidence", f"{prob_value:.2%}")
                                    st.progress(prob_value)
                        
                    except Exception as e:
                        st.error(f"Error analyzing text: {str(e)}")
                        st.info("This might happen if the text preprocessing doesn't match the training preprocessing.")
            else:
                st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()