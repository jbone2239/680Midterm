# student_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# Load the dataset
df = pd.read_csv(r'C:\Users\allib\OneDrive\Desktop\MS Data Science\ANA680\Week2\StudentsPerformance.csv')
 
df = df[['math score', 'reading score', 'writing score', 'race/ethnicity']]

# Encode target labels
le = LabelEncoder()
df['race/ethnicity_encoded'] = le.fit_transform(df['race/ethnicity'])

X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model 
model = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Save model and scalers
joblib.dump(model, 'student_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
