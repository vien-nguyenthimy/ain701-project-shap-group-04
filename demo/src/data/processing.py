import pandas as pd
from sklearn.model_selection import train_test_split

def drop_unnecessary_columns(df):
    # Loại bỏ ID vì không có giá trị dự báo
    if 'LoanID' in df.columns:
        df = df.drop(columns=['LoanID'])
    return df

def encode_features(df):
    """Thực hiện One-Hot Encoding cho các biến categorical."""
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

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