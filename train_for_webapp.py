import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("final_endometriosis_combined_dataset.csv")

# Target
y = df["Diagnosis"]
if y.dtype == object:
    y = y.map({"Yes":1,"No":0,"Positive":1,"Negative":0})
y = y.astype(int)

# Features
X = df.drop("Diagnosis", axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Test accuracy
preds = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(model, "models/ml_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ ML model saved successfully!")