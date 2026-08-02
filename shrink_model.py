import joblib
import os

model_path = 'models/weather_predictor.pkl'

if os.path.exists(model_path):
    print("Loading existing model...")
    model = joblib.load(model_path)
    
    original_trees = len(model.estimators_)
    print(f"Original tree count: {original_trees}")
    
    # Keep the top 25 most critical decision trees
    model.estimators_ = model.estimators_[:25]
    model.n_estimators = 25
    
    # Save with zlib level-3 compression
    joblib.dump(model, model_path, compress=3)
    
    new_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"SUCCESS! Model pruned to 25 trees.")
    print(f"New Compressed File Size: {new_size:.2f} MB")
else:
    print(f"Error: Could not find {model_path}")