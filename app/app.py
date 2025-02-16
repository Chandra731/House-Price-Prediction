import sys
import os
from flask import Flask, request, render_template
import joblib
import pandas as pd

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.predict import make_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = {k: [float(v)] if k != 'ocean_proximity' else [v] for k, v in data.items()}
    prediction = make_prediction(data)
    
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)