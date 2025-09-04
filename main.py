import streamlit as st
import joblib
import torch
#from transformers import BertTokenizer, BertForSequenceClassification

# ======================
# Load models
# ======================
@st.cache_resource
def load_models():
    svm_model = joblib.load("models/svm_model.pkl")
    nb_model = joblib.load("models/naive_bayes_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")

    # Labels used in Naive Bayes training
    target_labels = [
        'severe_toxicity_bin',
        'obscene_bin',
        'threat_bin',
        'insult_bin',
        'identity_attack_bin'
    ]

    return svm_model, nb_model, vectorizer, tokenizer, bert_model, target_labels


svm_model, nb_model, vectorizer, tokenizer, bert_model, target_labels = load_models()

# ======================
# Prediction functions
# ======================
def predict_svm(text):
    features = vectorizer.transform([text])
    return svm_model.predict(features)[0]

def predict_nb(text):
    features = vectorizer.transform([text])
    preds = nb_model.predict(features)[0]
    return {label: preds[i] for i, label in enumerate(target_labels)}

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# ======================
# Streamlit UI
# ======================
st.title("🧹 Toxic Comment Classifier")
st.write("Classify comments as **toxic** or **non-toxic** using SVM, Naive Bayes, and BERT models.")

user_input = st.text_area("Enter a comment to analyze:", "")

if st.button("Classify"):
    if user_input.strip():
        # Run predictions
        svm_pred = predict_svm(user_input)
        nb_pred = predict_nb(user_input)
        bert_pred = predict_bert(user_input)

        labels = {0: "Non-toxic", 1: "Toxic"}

        st.subheader("Results:")

        st.write(f"**SVM Prediction:** {labels[svm_pred]}")
        st.write(f"**BERT Prediction:** {labels[bert_pred]}")

        st.markdown("**Naive Bayes Predictions (multi-label):**")
        for label, val in nb_pred.items():
            st.write(f"- {label}: {labels[val]}")
    else:
        st.warning("Please enter a comment before classifying.")
