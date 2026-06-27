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

def get_models():
    """Trả về danh sách các mô hình cần huấn luyện"""
    models = {
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    return models


def train_single_model(model, X_train, y_train):
    """Huấn luyện 1 mô hình và đo thời gian"""
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    
    print(f"Training completed in {elapsed:.2f} seconds")
    return model, elapsed


def evaluate_model(model, X_test, y_test):
    """Đánh giá mô hình với các chỉ số"""
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
    
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    return metrics, cm, (fpr, tpr)


def train_all_models(X_train, y_train, X_test, y_test, verbose=True):
    """Huấn luyện tất cả mô hình và lưu kết quả"""
    models = get_models()
    results = {}
    roc_data = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Huấn luyện
        trained_model, elapsed = train_single_model(model, X_train, y_train)
        
        # Đánh giá
        metrics, cm, roc = evaluate_model(trained_model, X_test, y_test)
        
        # Lưu kết quả
        results[name] = {'model': trained_model, 'metrics': metrics, 'confusion_matrix': cm, 'training_time': elapsed}
        roc_data[name] = roc
        
        print(f"AUC={metrics['ROC-AUC']} | F1={metrics['F1-Score']} | Time={elapsed:.1f}s")
    print("Đã huấn luyện tất cả các mô hình")
    
    return results, roc_data


def get_results_summary(results):
    """"Tạo bảng so sánh các mô hình"""
    summary_data = []
    
    for name, result in results.items():
        metrics = result['metrics']
        summary_data.append({'Model': name, 'Accuracy': metrics['Accuracy'], 'Precision': metrics['Precision'], 'Recall': metrics['Recall'], 'F1-Score': metrics['F1-Score'], 'ROC-AUC': metrics['ROC-AUC'], 'Time (s)': round(result['training_time'], 2)})
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1-Score', ascending=False)
    
    return summary_df


def save_model(model, model_name, save_dir='src/models'):
    """Lưu mô hình ra file"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_path = save_path / f"{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")
    
    return file_path


def save_all_models(results, save_dir='src/models'):
    for name, result in results.items():
        save_model(result['model'], name, save_dir)
    
    print("\nĐã lưu tất cả các mô hình")

def save_results_summary(results, save_path='demo/src/reports/files/model_results.csv'):
    """Lưu kết quả đánh giá mô hình vào file CSV"""
    summary_df = get_results_summary(results)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    summary_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
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

    print(summary_df.to_string(index=False))
    
    # Save results
    save_results_summary(results)
    
    # Save models
    if save_models:
        save_all_models(results)
    
    # Get best model
    best_model_name, best_model, best_metrics = get_best_model(results)
    
    return results, summary_df, best_model_name, best_model, roc_data