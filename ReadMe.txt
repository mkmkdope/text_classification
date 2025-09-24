-- dataset
https://www.kaggle.com/datasets/julian3833/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv

/*
File Purpose:
main.py -> streamlit (UI) for all models to test the function.
NaiveBayes_SVM.ipynb -> train, save, evaluate for naive bayes and svm model.

Python Version:
NaiveBayes_SVM.ipynb -> 3.12.7
main.py -> 3.11.9
*/

-- follow the step below to create virtual environment (venv) for main.py
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run streamlit 
streamlit run main.py

# Deactivate
deactivate

-- Change to venv (VS Code)
Ctrl + Shift + P
Python: Select Interpreter

