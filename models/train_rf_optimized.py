import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    brier_score_loss,
    roc_auc_score
)

def train_gunupur_model():
    print("⏳ Initializing MLOps Training Pipeline for Gunupur Microclimate...")
    
    # 1. Locate and ingest the Gunupur dataset safely from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "processed", "gunupur_cleaned.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}. Run scripts/fetch_gunupur_data.py first.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} historical records from {data_path}")
    
    # 2. Enforce the exact 15-Feature Schema required by your frontend UI
    FEATURE_SCHEMA = [
        'RH2M', 'T2MDEW', 'QV2M', 'T2MWET', 'PS', 'PSC',
        'T2M_MAX', 'T2M_MIN', 'TS', 'ALLSKY_SFC_UV_INDEX',
        'WS50M', 'WD50M', 'WSC', 'LATITUDE', 'LONGITUDE'
    ]
    
    X = df[FEATURE_SCHEMA]
    y = df['Target_Rain']
    
    # 3. Stratified Train/Test Split (80% Train / 20% Unseen Validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Base Estimator: Memory-Optimized Random Forest
    # max_depth=12 prevents exponential node memory expansion in cloud RAM
    base_rf = RandomForestClassifier(
        n_estimators=35,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',  # Mathematically compensates for 32% rain imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Stratified 5-Fold Cross-Validation on Training Set
    print("⏳ Executing Stratified 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_rf, X_train, y_train, cv=cv, scoring='f1')
    print(f"📊 5-Fold CV Mean F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 6. Probability Calibration Layer (Isotonic Calibration)
    print("⏳ Fitting Calibrated Classifier for dependable percentage outputs...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_rf,
        method='sigmoid',
        cv=5
    )
    calibrated_model.fit(X_train, y_train)
    
    # Explicitly attach feature names so FastAPI /metadata discovery works
    calibrated_model.feature_names_in_ = np.array(FEATURE_SCHEMA)
    
    # 7. Out-of-Sample Performance Evaluation on Test Set
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    roc = roc_auc_score(y_test, y_prob)
    
    print("\n========================================================")
    print("          GUNUPUR MODEL PERFORMANCE DEFENSE SUMMARY      ")
    print("========================================================")
    print(f"  Out-of-Sample F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC Discriminability : {roc:.4f}")
    print(f"  Brier Calibration Score  : {brier:.4f} (Closer to 0.0 is better)")
    print("========================================================")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Rain (0)', 'Rain (1)']))
    
    # 8. Serialize and Export Production Binary
    output_model_path = os.path.join(project_root, "models", "weather_predictor.pkl")
    joblib.dump(calibrated_model, output_model_path, compress=3)
    
    file_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"\n✅ Successfully exported calibrated Gunupur model to: {output_model_path}")
    print(f"📦 Binary File Size: {file_size_mb:.2f} MB (Optimized for 512MB Cloud Containers)")

if __name__ == "__main__":
    train_gunupur_model()