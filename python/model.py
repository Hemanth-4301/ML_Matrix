from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from fuzzywuzzy import process  

app = Flask(__name__)

df = pd.read_csv('dataset.csv')  

symptom_encoder = OneHotEncoder(sparse_output=False)  
X = symptom_encoder.fit_transform(df[['symptom']])  
y = df['disease'].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}') 

@app.route('/predict', methods=['POST'])
def predict():
    symptom = request.json['symptom'].strip().lower()  
    print(f"Received symptom: {symptom}") 
    
    try:
        if symptom not in df['symptom'].values:  
            closest_match = process.extractOne(symptom, df['symptom'])
            matched_symptom, score = closest_match
            
            if score >= 80:
                symptom = matched_symptom
            else:
                return jsonify({'error': 'Symptom not found, please check your input.'}), 400
        
        symptom_encoded = symptom_encoder.transform([[symptom]])  
        prediction = model.predict(symptom_encoded)  
        
        disease = prediction[0]  
        medicine = df.loc[df['disease'] == disease, 'medicine'].values[0]  
        return jsonify({'disease': disease, 'medicine': medicine})
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400  

if __name__ == '__main__':
    app.run(debug=True, port=5001) 
