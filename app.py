from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load dataset
df = pd.read_csv("machine_data.csv")

# ---------------- DATA ANALYSIS ----------------
df['efficiency'] = (df['output_power'] / df['input_power']) * 100

df['health_score'] = (
    0.5 * df['efficiency']
    - 0.3 * df['temperature']
    - 0.2 * df['vibration'] * 100
)

# ---------------- AI MODEL ----------------
X = df[['temperature', 'vibration', 'load']]
y = df['efficiency']

model = RandomForestRegressor()
model.fit(X, y)

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.form

    input_power = float(data['input_power'])
    output_power = float(data['output_power'])
    temperature = float(data['temperature'])
    vibration = float(data['vibration'])
    load = float(data['load'])

    efficiency = (output_power / input_power) * 100

    health_score = (
        0.5 * efficiency
        - 0.3 * temperature
        - 0.2 * vibration * 100
    )

    predicted_eff = model.predict([[temperature, vibration, load]])[0]

    if health_score > 80:
        status = "Excellent"
    elif health_score > 60:
        status = "Warning"
    else:
        status = "Critical"

    return jsonify({
        "efficiency": round(efficiency, 2),
        "health_score": round(health_score, 2),
        "predicted_efficiency": round(predicted_eff, 2),
        "status": status
    })

if __name__ == '__main__':
    app.run(debug=True)
