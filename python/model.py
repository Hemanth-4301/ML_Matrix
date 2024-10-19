from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from fuzzywuzzy import process  # Import fuzzy matching

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('dataset.csv')  # Ensure the CSV file is in the same directory as this script

# One-hot encode the symptoms
symptom_encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name
X = symptom_encoder.fit_transform(df[['symptom']])  # Use lowercase 'symptom' for consistency
y = df['disease'].values  # Use lowercase 'disease' for consistency

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')  # Print the accuracy

@app.route('/predict', methods=['POST'])
def predict():
    symptom = request.json['symptom'].strip().lower()  # Convert input to lowercase
    print(f"Received symptom: {symptom}")  # Debugging line
    
    try:
        # Check if the symptom exists in the original dataframe
        if symptom not in df['symptom'].values:  # Use lowercase 'symptom'
            # Use fuzzy matching to find the closest symptom
            closest_match = process.extractOne(symptom, df['symptom'])
            matched_symptom, score = closest_match
            
            # If the match score is above a certain threshold (e.g., 80), use it
            if score >= 80:
                symptom = matched_symptom
            else:
                return jsonify({'error': 'Symptom not found, please check your input.'}), 400
        
        # Encode the symptom
        symptom_encoded = symptom_encoder.transform([[symptom]])  # Encode symptom
        prediction = model.predict(symptom_encoded)  # Make prediction
        
        # Get medicine suggestion based on the predicted disease
        disease = prediction[0]  # The predicted disease
        
        # Find corresponding medicine (ensure 'medicine' is in your dataset)
        medicine = df.loc[df['disease'] == disease, 'medicine'].values[0]  # Find corresponding medicine
        
        return jsonify({'disease': disease, 'medicine': medicine})
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400  # Return error response

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Start the Flask app
