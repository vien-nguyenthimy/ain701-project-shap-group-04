"""
Prediction & Preprocessing functions for single loan application
"""
import joblib
import pandas as pd
from pathlib import Path
import numpy as np

def get_project_root():
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent  # Trỏ về demo/


def load_preprocessor_and_model():
    """Load model và các thông tin preprocessing"""
    root = get_project_root()
    
    model_path = root / "data" / "models" / "lightgbm.pkl"
    model = joblib.load(model_path)
    
    # Load preprocessing info nếu có
    try:
        preprocessing_info = joblib.load(root / "data" / "processed" / "preprocessing_info.pkl")
    except:
        preprocessing_info = None
    
    try:
        feature_names = pd.read_csv(root / "data" / "processed" / "feature_names.txt", header=None).iloc[:,0].tolist()
    except:
        feature_names = None
    
    print(f"Loaded LightGBM model successfully!")
    return model, preprocessing_info, feature_names


def preprocess_single_input(user_input_dict, feature_names):
    """Chuyển input người dùng thành DataFrame đúng format"""
    df = pd.DataFrame([user_input_dict])
    
    # Đảm bảo có đủ features (thêm missing features = 0 hoặc giá trị mặc định)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # hoặc giá trị median/default phù hợp
    
    # Giữ đúng thứ tự features
    df = df[feature_names]
    return df


def predict_loan(user_input_dict):
    """
    Dự đoán + trả về xác suất + SHAP ready data
    user_input_dict: dictionary từ người dùng
    """
    model, preprocessing_info, feature_names = load_preprocessor_and_model()
    
    # Preprocess
    input_df = preprocess_single_input(user_input_dict, feature_names)
    
    # Dự đoán
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba > 0.5 else 0
    
    result = {
        'prediction': prediction,
        'probability_default': float(proba),
        'status': "Rủi ro cao (Default)" if prediction == 1 else "An toàn (Non-Default)",
        'input_data': input_df.iloc[0].to_dict()
    }
    
    return result, input_df