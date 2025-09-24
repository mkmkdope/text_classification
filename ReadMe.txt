-- dataset
https://www.kaggle.com/datasets/julian3833/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv

"""
bert.py -> Python version 3.13
NaiveBayes_SVM.ipynb -> Python version 3.12.7
main.py -> Python Version 3.11.9
"""

-- follow the step below to create virtual environment (venv) for main.py
# Create virtual environment
py -3.11 -m venv venv
python -m venv venv

# Activate
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install torch & transformers
pip install transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Run streamlit 
streamlit run main.py

# Deactivate
deactivate

-- Change to venv (VS Code)
Ctrl + Shift + P
Python: Select Interpreter

/*
File Purpose:
main.py -> streamlit (UI) for all models to test the function.
NaiveBayes_SVM.ipynb -> train, save, evaluate for naive bayes and svm model.
bert.py -> train, save, evaluate for BERT model.
*/