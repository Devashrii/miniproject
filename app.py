from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Naive Bayes model
model_path = 'naive_bayes_model.pkl'
model = pickle.load(open(model_path, 'rb'))

# Define the features for the input form
features = [
    "Age", "Fast_Food_Intake", "Physical_Activity", "Sugar_Intake_Per_Day",
    "Fiber_Intake_Per_Day", "Protein_Intake_Per_Day", "Processed_Food_Score",
    "Dietary_Pattern", "Socioeconomic_Status", "Environmental_Factors",
    "Region", "Industrialization_Level", "Nutrition_Access"
]

# Categorical features for dropdown menus
categorical_features = {
    "Dietary_Pattern": ["Pattern_1", "Pattern_2", "Pattern_3"],  # Replace with actual categories
    "Socioeconomic_Status": ["Low", "Middle", "High"],
    "Environmental_Factors": ["Factor_1", "Factor_2", "Factor_3"],
    "Region": ["Region_1", "Region_2", "Region_3"],
    "Industrialization_Level": ["Low", "Medium", "High"],
    "Nutrition_Access": ["Limited", "Moderate", "Abundant"]
}

# Load label encoders to map categorical values
label_encoders_path = 'naive_bayes_model.pkl'
label_encoders = pickle.load(open(label_encoders_path, 'rb'))

@app.route('/')
def index():
    """Render the prediction form."""
    return render_template('index.html', features=features, categorical_features=categorical_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions."""
    try:
        # Collect input data
        input_data = {}
        for feature in features:
            if feature in categorical_features:  # Handle categorical inputs
                value = request.form.get(feature)
                input_data[feature] = label_encoders[feature].transform([value])[0]
            else:  # Handle numerical inputs
                input_data[feature] = float(request.form.get(feature))
        
        # Convert input to DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        # Interpret prediction
        result = "Early" if prediction == 1 else "Late"

        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)