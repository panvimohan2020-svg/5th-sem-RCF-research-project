import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys

# 1. Load your dataset using the verified raw string path
file_path = r'C:\Users\lenovo\OneDrive\Documents\5th sem RCF research project\Rainfall_Classification_Project\Data\Raw\Processed'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"CRITICAL ERROR: Could not find the file at {file_path}")
    sys.exit(1)

# 2. Dynamically identify the target column
possible_targets = ['Rain', 'Rainfall', 'rain', 'Rain_Today', 'Precipitation', 'Target', 'Class']
target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

# Fallback: Assume the last column is the target if standard names aren't found
if not target_column:
    target_column = df.columns[-1]
    print(f"WARNING: Standard target name not found. Defaulting to the last column: '{target_column}'")
else:
    print(f"SUCCESS: Identified target column as: '{target_column}'")

# 3. Isolate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# XGBoost requires the target variable to be numeric (0 and 1)
# We use LabelEncoder in case your target contains strings like 'Yes'/'No'
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) 

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Initialize XGBoost with strict memory and depth caps for the cloud
model = XGBClassifier(
    n_estimators=100,       
    max_depth=5,            
    learning_rate=0.1,      
    random_state=42,
    eval_metric='logloss'
)

# 6. Train the model
print("Training XGBoost model...")
model.fit(X_train, y_train)

# 7. Validate Accuracy
y_pred = model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 8. Export the lightweight model with maximum compression
joblib.dump(model, 'weather_predictor.pkl', compress=3)
print("Optimized model saved successfully.")