import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Define the filename for the exported model
filename = 'logistic_regression_model.joblib'

# Save the trained model to the file
joblib.dump(model, filename)

print(f"Model successfully exported to {filename}")

# Define the filename for the exported scaler
scaler_filename = 'scaler.joblib'

# Save the trained scaler to the file
joblib.dump(scaler, scaler_filename)

print(f"StandardScaler successfully exported to {scaler_filename}")

# Load the pre-trained logistic regression model and scaler
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define the expected columns based on the training data (from the kernel state for X)
# This is crucial for ensuring the input data for the API has the correct structure.
expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or Scaler not loaded'}), 500

    try:
        # Get JSON data from the request body
        data = request.get_json(force=True)

        # Convert incoming JSON data into a pandas DataFrame
        # It's important that the input data matches the structure expected by the model.
        # If a single prediction is expected, wrap the data in a list to create a DataFrame row.
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input data format, expected dict or list of dicts.'}), 400

        # Ensure the DataFrame has the correct column order and names
        # Reindex to ensure all expected columns are present and in the correct order.
        # Fill missing columns with 0 or a suitable default if necessary, though it's better to ensure client sends all data.
        input_df = input_df[expected_columns]

        # Use the loaded StandardScaler to transform the new data
        scaled_data = scaler.transform(input_df)

        # Make a prediction using the loaded logistic regression model
        prediction = model.predict(scaled_data)

        # Convert prediction to human-readable format
        result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

        # Return the prediction as a JSON response
        return jsonify({'prediction': result})

    except KeyError as e:
        return jsonify({'error': f'Missing expected feature in input data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask application
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, host='0.0.0.0', port=5000)
