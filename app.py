"""Flask application for predicting mean salary using median salary,
area size, number of jobs and population size."""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

def load_model_and_scaler():
    """Load the pre-trained model and scaler."""
    loaded_model = joblib.load('models/housing_price_model.pkl')
    loaded_scaler = joblib.load('models/housing_scaler.pkl')
    return loaded_model, loaded_scaler

# Load model and scaler once when the application starts
model, scaler = load_model_and_scaler()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the form data.

    Returns:
        JSON response containing the predicted salary.
    """
    # Extract input data from the request form
    input_features = [
        float(request.form['median_salary']),
        float(request.form['population_size']),
        float(request.form['number_of_jobs']),
        float(request.form['area_size'])
    ]

    # Convert the input data to a numpy array
    new_data = np.array([input_features])

    # Scale the data using the scaler
    scaled_data = scaler.transform(new_data)

    # Make a prediction using the model
    predicted_salary = model.predict(scaled_data)[0]

    # Return the prediction as a JSON response
    return jsonify({'predicted_salary': predicted_salary})

if __name__ == '__main__':
    app.run(debug=True)
