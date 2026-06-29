"""
Model training and evaluation functions for loan default prediction.
"""
from pathlib import Path  
import joblib  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import json

def get_project_root():
    """Lấy project root một cách an toàn"""
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # demo/src/models → demo
    except:
        # Trường hợp chạy từ notebook hoặc trực tiếp
        project_root = Path.cwd().parent if Path.cwd().name == "demo" else Path.cwd()
    
    return project_root

def get_models(y_train):
    """Trả về danh sách các mô hình cần huấn luyện"""
    neg = (y_train == 0).sum()  # Số lượng non-default
    pos = (y_train == 1).sum()  # Số lượng default
    scale_pos_weight = neg / pos
    print(f"Class 0 (Non-Default): {neg:,}")
    print(f"Class 1 (Default): {pos:,}")
    print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

    models = {
        'XGBoost': XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, eval_metric='logloss', random_state=42, n_jobs=-1, scale_pos_weight = scale_pos_weight),
        'LightGBM': LGBMClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1, is_unbalance=True),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
    }

    return models

def train_single_model(model, X_train, y_train):
    """Huấn luyện 1 mô hình và đo thời gian"""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training {type(model).__name__} completed in {elapsed:.2f} seconds")
    return model, elapsed


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    
    return metrics, confusion_matrix(y_test, y_pred), roc_curve(y_test, y_prob)


def train_all_models(X_train, y_train, X_test, y_test, verbose=True):
    """Huấn luyện tất cả mô hình và lưu kết quả"""
    models = get_models(y_train)
    results = {}
    roc_data = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_model, elapsed = train_single_model(model, X_train, y_train)
        metrics, cm, roc = evaluate_model(trained_model, X_test, y_test)
        results[name] = {'model': trained_model, 'metrics': metrics, 'confusion_matrix': cm, 'training_time': elapsed}
        roc_data[name] = roc
        
        print(f"AUC={metrics['ROC-AUC']} | F1={metrics['F1-Score']} | Time={elapsed:.1f}s")
    
    print("\nĐã huấn luyện tất cả các mô hình")
    
    return results, roc_data


def get_results_summary(results):
    """"Tạo bảng so sánh các mô hình"""
    summary_data = []
    
    for name, result in results.items():
        metrics = result['metrics']
        summary_data.append({'Model': name, 
                             'Accuracy': metrics['Accuracy'], 
                             'Precision': metrics['Precision'], 
                             'Recall': metrics['Recall'], 
                             'F1-Score': metrics['F1-Score'], 
                             'ROC-AUC': metrics['ROC-AUC'], 
                             'Time (s)': round(result['training_time'], 2)
        })
    
    df = pd.DataFrame(summary_data)
    return df.sort_values('F1-Score', ascending=False)


def save_model(model, model_name):
    project_root = get_project_root()
    save_dir = project_root / "data" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, file_path)
    return file_path


def save_all_models(results):
    """Lưu tất cả models"""
    for name, result in results.items():
        save_model(result['model'], name)
    
    print("\nĐã lưu tất cả các mô hình")


def save_results_summary(results, save_path=None):
    """Lưu bảng so sánh kết quả"""
    project_root = get_project_root()
    
    if save_path is None:
        save_path = project_root / 'src' / 'reports' / 'model_results.csv'
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    summary_df = get_results_summary(results)
    summary_df.to_csv(save_path, index=False)

    return summary_df


def get_best_model(results, metric='F1-Score'):
    """Tìm mô hình tốt nhất dựa trên metric"""
    summary_df = get_results_summary(results)
    best_row = summary_df.iloc[0] #Dòng đầu tiên
    
    best_model_name = best_row['Model']
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]['metrics']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"   {metric}: {best_metrics[metric]}")
    
    return best_model_name, best_model, best_metrics


def train_and_save_pipeline(X_train, y_train, X_test, y_test, save_models=True):
    """Chạy pipeline huấn luyện và lưu mô hình"""
    # Chạy tất cả các mô hình trong 1 lần
    results, roc_data = train_all_models(X_train, y_train, X_test, y_test)
    
    #Tạo bảng so sánh
    summary_df = get_results_summary(results)
    
    # Save results
    save_results_summary(results)
    
    # Save models
    if save_models:
        save_all_models(results)
    
    return results, summary_df, roc_data