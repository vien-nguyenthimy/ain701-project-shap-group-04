import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

def drop_unnecessary_columns(df):
    # Loại bỏ ID vì không có giá trị dự báo
    if 'LoanID' in df.columns:
        df = df.drop(columns=['LoanID'])
    return df

def encode_features(df):
    """Thực hiện One-Hot Encoding cho các biến categorical."""
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    print(f"Encoded {len(categorical_cols)} categorical columns")
    return df_encoded


def encode_single_input(df, feature_names):
    """Encode 1 dòng input cho predict"""
    row = {col: 0 for col in feature_names}
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        value = df.iloc[0][col]
        dummy_col = f"{col}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col in row:
            row[col] = df.iloc[0][col]
    
    return pd.DataFrame([row])[feature_names]


def get_features_target(df_encoded, target_col='Default'):
    """Tách ma trận đặc trưng X và biến mục tiêu y."""
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Chia dữ liệu thành tập Train và Test với tính năng phân tầng (stratify)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test
