# Create virtual environment
python -m venv myenv

# Activate on Windows
myenv\Scripts\activate

# Then install requirements
pip install -r requirements.txt

streamlit run main.py

# dataset
# https://www.kaggle.com/datasets/julian3833/jigsaw-unintended-bias-in-toxicity-classification/data?select=train.csv

# change to venv
# Ctrl + Shift + P (Windows/Linux)
# Python: Select Interpreter