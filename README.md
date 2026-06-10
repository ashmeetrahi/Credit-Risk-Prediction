# Credit Risk Prediction Model

## Project Overview
Developed an end-to-end machine learning pipeline using a **Random Forest Classifier** to predict loan default risks. The project involved rigorous data cleaning, handling missing values, encoding categorical variables, and evaluating real-world business metrics on a large-scale credit dataset.

## Key Performance Metrics
* **Model Accuracy:** 92.81%
* **Defaulter Precision (Class 1):** 95% (High reliability in flagging risky borrowers)
* **Defaulter F1-Score:** 0.81
* **Total Evaluated Records:** 28,632 rows (after data cleaning)

## Feature Importance Rankings
The model analyzed data patterns to determine the primary drivers of loan default risk:
1. **Loan-to-Income Percentage:** 22.11%
2. **Individual Income:** 14.09%
3. **Interest Rate:** 12.75%
4. **Loan Grade:** 11.27%
5. **Home Ownership Status:** 9.69%

## Tech Stack & Libraries Used
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn (RandomForestClassifier, LabelEncoder, train_test_split)
