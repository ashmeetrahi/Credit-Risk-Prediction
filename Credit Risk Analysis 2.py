import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
data_path = '/Users/ashmeetrahi/downloads/credit_risk_dataset.csv'  # Simplified path for portability
df = pd.read_csv(data_path)

print("--- Data Initial Inspection ---")
print("Top 5 Oldest Ages:\n", df['person_age'].sort_values(ascending=False).head())

# 2. Data Cleaning & Preprocessing
# Identify impossible data points where employment length exceeds age
impossible_work = df[df['person_emp_length'] > df['person_age']]
print(f"\nRows with conflicting work history: {len(impossible_work)}")

# Filter out anomalies and handle missing values
df = df[(df['person_age'] < 100) & (df['person_emp_length'] <= df['person_age'])]
df = df.dropna()  # Ensures no missing records are passed to the model

# 3. Categorical Encoding
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

print("\nEncoded Dataset Sample:\n", df.head())




# 4. Feature Engineering & Train-Test Split
X = df.drop('loan_status', axis=1) 
y = df['loan_status']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
print("\n--- Training Model ---")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

# 6. Evaluation and Inference
y_pred = rf_model.predict(X_test)

print('\nFirst 10 Predictions:', y_pred[:10])
print("Total Evaluated Rows:", df.shape[0])

# 7. Performance Metrics
score = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {score * 100:.2f}%")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Feature Importance Analysis
importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance Rankings:\n", importance_df)


