from flask import Flask, request, jsonify, render_template  # <--- add render_template
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

EXPECTED_COLUMNS = [
    'Age','Sex','ChestPainType','RestingBP','Cholesterol',
    'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'
]

@app.route('/')
def home():
    return render_template('index.html')   # <--- Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if not all(col in data for col in EXPECTED_COLUMNS):
        return jsonify({'error': f'All columns required: {EXPECTED_COLUMNS}'}), 400
    try:
        values = [float(data[col]) for col in EXPECTED_COLUMNS]
        sample = np.array(values).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'Error parsing input: {str(e)}'}), 400
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1]) if hasattr(model, "predict_proba") else None
    result_message = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "result": result_message
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)
