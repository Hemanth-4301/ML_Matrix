# Medicine-Recommendation-system
Your project is a Medicine Recommendation System built with Flask and machine learning. It allows users to input symptoms and receive predictions for likely diseases along with recommended medicines. Flask provides the web interface for users to submit symptoms and get results, while a machine learning model handles the prediction process.

The system uses a dataset (dataset.csv) containing symptoms, diseases, and medicines. Since machine learning models work with numbers, the symptoms are transformed into a numerical format using OneHotEncoder, which converts each symptom into a binary vector of 0s and 1s. This encoding allows the model to interpret the symptoms as features for training.

The data is split into two parts using train_test_split: 80% for training the model and 20% for testing it. This ensures the model learns effectively and can be evaluated on unseen data. A Random Forest Classifier is used as the prediction model. It creates multiple decision trees, each making its own prediction, and combines them for a final, more accurate result.

The system is user-friendly, handling input via Flask’s POST route, where users submit symptoms to get a disease and medicine recommendation. It also incorporates fuzzy matching to correct minor input errors, ensuring flexibility in symptom entry. Overall, the project provides a helpful interface for diagnosing diseases and recommending treatments based on symptoms.
