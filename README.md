🩺 Disease and Medicine Predictor

An Intelligent Medical Assistant for Accurate Disease Diagnosis and Medicine Suggestions

🌟 About the Project
The Disease and Medicine Predictor is a machine learning-powered web application designed to make healthcare more accessible. Simply input your symptoms, and the system predicts potential diseases while recommending appropriate medicines—all through an intuitive and user-friendly interface.

🔍 Key Highlights:

Accurate Predictions: Uses a trained Random Forest Classifier for disease detection.
Medicine Recommendations: Suggests treatments tailored to the predicted disease.
User-Friendly: Handles minor input errors using fuzzy matching.

🛠️ Technologies Used
Python
React.js
Flask
Data Processing	Pandas-OneHotEncoder
Machine Learning-scikit-learn (Random Forest Classifier)
Utilities	FuzzyWuzzy for input correction

🎯 Features
🖥️ Web-Based: Access predictions through a responsive web interface.
🔄 Real-Time: Get disease predictions instantly with suggested treatments.
🤝 Robust Input Handling: Fixes minor errors in symptom entry with fuzzy matching.
📊 Machine Learning Backend: Trained on a comprehensive dataset for high accuracy.

📚 How It Works
Users enter symptoms in the input field (e.g., fever, headache, nausea).
The system transforms these symptoms into machine-readable vectors using OneHotEncoder.
A Random Forest Classifier predicts the most probable disease.
Medicines for the predicted disease are displayed to the user.
