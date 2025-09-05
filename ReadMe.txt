# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Then install requirements
pip install -r requirements.txt

streamlit run main.py

# dataset
# https://www.kaggle.com/datasets/julian3833/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv

# change to venv (VS Code)
# Ctrl + Shift + P
# Python: Select Interpreter