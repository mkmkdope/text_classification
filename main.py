# main.py
# Streamlit app that unifies SVM / Naive Bayes (.pkl) and BERT (HF folder) on one page

import os
import json
import joblib
import numpy as np
import streamlit as st

# ------------ (A) A tiny BERT wrapper so HF behaves like scikit models ------------
# Note on dependencies:
# - SVM / Naive Bayes run with scikit-learn only (no torch required).
# - BERT requires `torch` and `transformers`. We import them lazily so the app
#   still runs without these packages (you can use SVM/NB only).

# Streamlit page config must be the first Streamlit call
st.set_page_config(page_title="Toxicity Classifier (BERT / SVM / NB)", page_icon="🤖", layout="centered")

class BertToxicModel:
    """
    Make BERT look like a scikit model:
      - predict(texts)       -> ndarray [N, L] of 0/1
      - predict_proba(texts) -> ndarray [N, L] of probabilities
      - labels               -> list of label names in the right order (pretty names)
    Loading path: export_dir (folder containing hf/, best_thresholds.json, label_columns.txt)
    """
    def __init__(self, export_dir="./models/bert", device=None, max_length=256):
        # Lazy import heavy deps so Streamlit can load without torch installed
        try:
            import torch as _torch
            from transformers import (
                AutoTokenizer as _AutoTokenizer,
                AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
            )
        except Exception as e:
            raise ImportError(
                "BERT dependencies missing. Install with: \n"
                "  python -m pip install transformers && \n"
                "  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \n"
                "(or choose a CUDA index-url if you have NVIDIA GPU).\n"
                f"Original error: {e}"
            ) from e

        self.torch = _torch
        self._AutoTokenizer = _AutoTokenizer
        self._AutoModelForSequenceClassification = _AutoModelForSequenceClassification
        self.export_dir = export_dir
        self.hf_dir = os.path.join(export_dir, "hf")
        if not os.path.isdir(self.hf_dir):
            raise FileNotFoundError(f"HuggingFace folder not found: {self.hf_dir}")

        self.device = self.torch.device(device or ("cuda" if self.torch.cuda.is_available() else "cpu"))
        # label order (training-time)
        lbl_path = os.path.join(export_dir, "label_columns.txt")
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(f"Missing label_columns.txt at: {lbl_path}")
        with open(lbl_path) as f:
            self.label_columns = [ln.strip() for ln in f if ln.strip()]

        # thresholds (optional; default 0.5 each)
        thr_path = os.path.join(export_dir, "best_thresholds.json")
        if os.path.exists(thr_path):
            with open(thr_path) as f:
                thr_dict = json.load(f)
            self.thresholds = np.array([thr_dict.get(k, 0.5) for k in self.label_columns], dtype=np.float32)
        else:
            self.thresholds = np.array([0.5] * len(self.label_columns), dtype=np.float32)

        # tokenizer & model
        self.tokenizer = self._AutoTokenizer.from_pretrained(self.hf_dir)
        self.model = self._AutoModelForSequenceClassification.from_pretrained(self.hf_dir)
        self.model.to(self.device).eval()

        # optional config
        self.max_length = max_length
        cfg_path = os.path.join(export_dir, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                self.max_length = int(cfg.get("max_length", max_length))
            except Exception:
                pass

    @property
    def labels(self):
        # pretty display names
        return [c.replace("_binary", "").replace("identity_attack", "identity_hate")
                for c in self.label_columns]

    def _forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits
            probs = self.torch.sigmoid(logits).cpu().numpy()  # [N, L]
        return probs

    def predict_proba(self, texts):
        return self._forward(texts)

    def predict(self, texts):
        probs = self._forward(texts)
        preds = (probs >= self.thresholds[None, :]).astype(int)
        return preds


# ------------ (B) Utilities for classic models (SVM / NB) ------------
def preprocess_text_minimal(text: str) -> str:
    # Keep minimal to avoid mismatch with teammates' vectorizer
    return text if text is None else str(text)

def _extract_scikit_proba(pred_proba):
    """
    Support multiple shapes:
      - list of length L, each (N,2) -> take [:,1]
      - ndarray (N, L)  -> already positive-class probs
    """
    if pred_proba is None:
        return None
    if isinstance(pred_proba, list):
        # MultiOutputClassifier style: list of arrays (N,2)
        pos = [pp[:, 1] if pp.ndim == 2 and pp.shape[1] >= 2 else pp.squeeze() for pp in pred_proba]
        return np.stack(pos, axis=1)  # (N, L)
    proba = np.asarray(pred_proba)
    if proba.ndim == 3 and proba.shape[2] == 2:
        # (L, N, 2) sometimes; make it (N, L)
        proba = proba[:, :, 1].T
    return proba  # expect (N, L)

def _extract_scikit_decision(dec):
    """Unify decision_function outputs to (N, L)."""
    if dec is None:
        return None
    if isinstance(dec, list):
        arrs = [np.asarray(d).reshape(-1) for d in dec]
        return np.stack(arrs, axis=1)
    arr = np.asarray(dec)
    if arr.ndim == 1:
        # (L,) -> (1, L)
        arr = arr[None, :]
    return arr

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _infer_label_names_by_length(L):
    # fallbacks if teammate didn't provide a label list file
    if L == 5:
        return ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    if L == 4:
        return ['obscene', 'identity_attack', 'insult', 'threat']
    # worst-case generic names
    return [f"label_{i}" for i in range(L)]

def pretty_name(name):
    return name.replace("_binary", "").replace("identity_attack", "identity_hate")

# ------------ (C) Cache loaders ------------
@st.cache_resource
def load_scikit_models(vec_path, nb_path, svm_path):
    vec = joblib.load(vec_path)
    models = {}
    if os.path.exists(nb_path):
        models['Naive Bayes'] = joblib.load(nb_path)
    if os.path.exists(svm_path):
        models['SVM'] = joblib.load(svm_path)
    return vec, models

@st.cache_resource
def load_bert_model(bert_dir):
    return BertToxicModel(export_dir=bert_dir)

# ------------ (D) Unified prediction ------------
def predict_unified(text, model, model_name, vectorizer=None):
    """
    Returns:
      {
        "is_toxic": bool,
        "overall_prob": float | None,
        "breakdown": {label: {"prediction": bool, "probability": float|None}}
      }
    """
    if model_name == "BERT":
        probs = model.predict_proba([text])[0]        # (L,)
        preds = model.predict([text])[0].astype(int)  # (L,)
        labels = model.labels
        overall = bool(preds.sum() > 0)
        breakdown = {}
        for i, name in enumerate(labels):
            breakdown[name] = {
                "prediction": bool(preds[i]),
                "probability": float(probs[i])
            }
        # overall probability: union = 1 - Π(1 - p_i)
        overall_prob = float(1.0 - float(np.prod(1.0 - probs)))
        return {"is_toxic": overall, "overall_prob": overall_prob, "breakdown": breakdown}

    # Classic scikit path
    X = vectorizer.transform([preprocess_text_minimal(text)])
    y_pred = model.predict(X)
    if hasattr(y_pred, "toarray"):
        y_pred = y_pred.toarray()
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:  # shape (L,)
        y_pred = y_pred[None, :]

    L = y_pred.shape[1]
    labels = _infer_label_names_by_length(L)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            proba = _extract_scikit_proba(proba)
        except Exception:
            proba = None

    breakdown = {}
    for i, raw_name in enumerate(labels):
        disp = pretty_name(raw_name)
        p = float(proba[0, i]) if (proba is not None) else None
        breakdown[disp] = {"prediction": bool(y_pred[0, i]), "probability": p}

    overall = any(v["prediction"] for v in breakdown.values())
    # if we do have per-label probabilities, we can compute a soft overall prob
    overall_prob = None
    if proba is not None:
        overall_prob = float(1.0 - float(np.prod(1.0 - proba[0, :])))

    return {"is_toxic": overall, "overall_prob": overall_prob, "breakdown": breakdown}


# ------------ (E) Streamlit UI ------------

st.title("🤖 Unified Toxicity Classifier")
st.caption("Compare BERT with traditional models on the same page.")

# Paths in sidebar (so your teammates can point to their files)
st.sidebar.header("Model Paths")
default_models_dir = "./models"
vec_path = st.sidebar.text_input("Vectorizer (.pkl)", os.path.join(default_models_dir, "vectorizer.pkl"))
nb_path  = st.sidebar.text_input("Naive Bayes (.pkl)", os.path.join(default_models_dir, "naive_bayes_model.pkl"))
svm_path = st.sidebar.text_input("SVM (.pkl)", os.path.join(default_models_dir, "svm_model.pkl"))
bert_dir = st.sidebar.text_input("BERT bundle folder", os.path.join(default_models_dir, "bert"))

st.sidebar.markdown("---")
st.sidebar.write("**Tips**")
st.sidebar.write("- BERT folder must contain `hf/` and `label_columns.txt` etc.")
st.sidebar.write("- SVM/NB should be trained with the same vectorizer.")

# Load models
vectorizer, sk_models = None, {}
load_errs = []
try:
    vectorizer, sk_models = load_scikit_models(vec_path, nb_path, svm_path)
except Exception as e:
    load_errs.append(f"Scikit models not fully loaded: {e}")

bert_model = None
try:
    bert_model = load_bert_model(bert_dir)
except Exception as e:
    st.warning(f"BERT not loaded ({e}). You can still use SVM/NB.")
else:
    sk_models["BERT"] = bert_model  # add bert into same dict for unified selection

# Model selection
available_names = list(sk_models.keys())
if not available_names:
    st.error("No models available. Check your paths in the sidebar.")
    st.stop()

model_descriptions = {
    "BERT": "Transformer model with contextual understanding (HuggingFace).",
    "SVM": "Strong linear classifier for high-dimensional text.",
    "Naive Bayes": "Fast probabilistic baseline for text classification."
}
selected = st.selectbox("Choose a model", available_names, index=available_names.index("BERT") if "BERT" in available_names else 0)
st.caption(model_descriptions.get(selected, ""))

# Warn if SVM/NB lacks calibrated probabilities
if selected != "BERT":
    _mdl = sk_models[selected]
    if not getattr(_mdl, "predict_proba", None) and getattr(_mdl, "decision_function", None):
        st.caption("Note: showing uncalibrated probabilities from decision_function (sigmoid).")

# Text input
text = st.text_area("Enter a comment to analyze:", height=140, placeholder="Type any sentence...")

# Predict
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()
    model = sk_models[selected]
    result = predict_unified(text, model, selected, vectorizer if selected != "BERT" else None)

    # Overall card
    is_toxic = result["is_toxic"]
    overall_prob = result["overall_prob"]
    if is_toxic:
        st.success("Overall: TOXIC")
    else:
        st.info("Overall: NON-TOXIC")
    if overall_prob is not None:
        st.write(f"Overall probability (soft union): **{overall_prob:.3f}**")

    # Show breakdown
    breakdown = result["breakdown"]
    if is_toxic:
        # Show only triggered sub-labels first (then optional non-triggered in expander)
        positives = {k: v for k, v in breakdown.items() if v["prediction"]}
        negatives = {k: v for k, v in breakdown.items() if not v["prediction"]}

        if positives:
            st.subheader("Triggered sub-labels")
            pos_rows = []
            for name, item in positives.items():
                prob = item["probability"]
                pos_rows.append([name, "TOXIC", f"{prob:.3f}" if prob is not None else "—"])
            st.table(
                {"Label": [r[0] for r in pos_rows],
                 "Prediction": [r[1] for r in pos_rows],
                 "Probability": [r[2] for r in pos_rows]}
            )

        with st.expander("See all sub-label scores"):
            all_rows = []
            for name, item in breakdown.items():
                prob = item["probability"]
                all_rows.append([name, "TOXIC" if item["prediction"] else "NON-TOXIC",
                                 f"{prob:.3f}" if prob is not None else "—"])
            st.table(
                {"Label": [r[0] for r in all_rows],
                 "Prediction": [r[1] for r in all_rows],
                 "Probability": [r[2] for r in all_rows]}
            )
    else:
        # NON-TOXIC: keep it simple, show a compact table with overall only;
        # optional: let user expand to see sub-label probabilities.
        with st.expander("Show sub-label probabilities"):
            all_rows = []
            for name, item in breakdown.items():
                prob = item["probability"]
                all_rows.append([name, "NON-TOXIC", f"{prob:.3f}" if prob is not None else "—"])
            st.table(
                {"Label": [r[0] for r in all_rows],
                 "Prediction": [r[1] for r in all_rows],
                 "Probability": [r[2] for r in all_rows]}
            )

# Footer
st.markdown("---")
st.caption("© Your Team — BERT + SVM + Naive Bayes")
