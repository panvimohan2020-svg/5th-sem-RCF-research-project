import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys

# 1. Load the dataset
file_path = r'C:\Users\lenovo\OneDrive\Documents\5th sem RCF research project\Rainfall_Classification_Project\Data\Raw\Rainfall_India.csv\UP_rainfall_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("CRITICAL ERROR: CSV file not found.")
    sys.exit(1)

# 2. Identify target column dynamically
target_column = None
for col in ['Rain', 'Rainfall', 'rain', 'Rain_Today', 'Precipitation', 'Target', 'Class']:
    if col in df.columns:
        target_column = col
        break
if not target_column: 
    target_column = df.columns[-1]

# 3. Separate features and target
X = df.drop(columns=[target_column]).copy()
y = df[target_column]

# 4. ENCODING FIX: Convert any string/categorical feature columns in X to numbers
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target variable
y_encoded = LabelEncoder().fit_transform(y.astype(str))

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Train Constrained Random Forest Model (Resource-Optimized for Cloud)
model = RandomForestClassifier(
    n_estimators=50,       # Constrained to 50 trees to keep .pkl size under 2MB
    max_depth=5,           # Depth limit to prevent tree bloat
    n_jobs=1,              # Single-threaded execution for free-tier memory safety
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Total Features Trained: {X.shape[1]}")
print(f"Constrained RF Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 7. Export micro-model
joblib.dump(model, 'weather_predictor.pkl', compress=3)
print("Micro-model saved successfully.")