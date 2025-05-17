from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# load model, scaler, and label encoder
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get user input from form
        math = float(request.form['math'])
        reading = float(request.form['reading'])
        writing = float(request.form['writing'])

        # format input as DataFrame to avoid feature name warning
        input_df = pd.DataFrame([[math, reading, writing]],
                                columns=['math score', 'reading score', 'writing score'])
        input_scaled = scaler.transform(input_df)

        # make prediction
        pred_class = model.predict(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        return render_template('index.html',
                               prediction_text=f"Predicted race/ethnicity: {pred_label}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
