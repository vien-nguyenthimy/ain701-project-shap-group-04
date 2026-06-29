"""
Prediction & Preprocessing functions for single loan application
"""
import joblib
import pandas as pd
from pathlib import Path
from src.data.processing import encode_single_input

THRESHOLD = 0.171


def get_project_root():
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def load_model_and_features(model_name='xgboost'):
    root = get_project_root()
    model_path = root / "data" / "models" / f"{model_name}.pkl"
    model = joblib.load(model_path)

    with open(root / "data" / "processed" / "feature_names.txt", encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    print(f"Loaded: {type(model).__name__} from {model_path.name}")
    print(f"Features: {len(feature_names)}")
    return model, feature_names


def predict_loan(user_input_dict, threshold=THRESHOLD):
    """Dự đoán + trả về xác suất + SHAP-ready data."""
    model, feature_names = load_model_and_features()

    df_raw = pd.DataFrame([user_input_dict])
    df_encoded = encode_single_input(df_raw, feature_names)

    proba = float(model.predict_proba(df_encoded)[0, 1])
    prediction = 1 if proba >= threshold else 0

    result = {
        'prediction': prediction,
        'probability_default': proba,
        'risk_level': 'High' if proba > 0.7 else 'Medium' if proba > 0.3 else 'Low',
        'status': "Rủi ro cao (Default)" if prediction == 1 else "An toàn (Non-Default)",
        'threshold_used': threshold,
        'input_data': df_encoded.iloc[0].to_dict()
    }
    return result, df_encoded