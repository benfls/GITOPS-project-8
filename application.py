from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

model = joblib.load('artifacts/models/model.pkl')
scaler = joblib.load('artifacts/processed/scaler.pkl')

FEATURES = ['Operation_Mode', 'Temperature_C', 'Vibration_Hz',
       'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
       'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
       'Predictive_Maintenance_Score', 'Error_Rate_%', 'Month', 'Day', 'Year', 'Hour'
]

LABELS = {
    0 : "Hight",
    1 : "Low",
    2 : "Medium"
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction = None

    if request.method == 'POST':
        try:
            input_feature = [float(request.form[feature]) for feature in FEATURES]
            input_array = np.array(input_feature).reshape(1, -1)

            scaled_array = scaler.transform(input_array)
            pred = model.predict(scaled_array)[0]
            prediction = LABELS.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')