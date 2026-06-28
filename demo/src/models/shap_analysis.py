"""
SHAP/TreeSHAP analysis functions for explaining loan default predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def get_explainer(model):
    """Tạo TreeExplainer cho mô hình tree-based (XGBoost/LightGBM)."""
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X, sample_size=None, random_state=42):
    """Tính SHAP values cho X. Nếu sample_size được set, lấy mẫu ngẫu nhiên
    để tăng tốc (test set 51k dòng không cần tính hết). Trả về (X_sample, shap_values)."""
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(sample_size, random_state=random_state).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)
    shap_values = explainer.shap_values(X_sample)
    return X_sample, shap_values


def plot_global_summary(shap_values, X_sample, save_path=None, max_display=15):
    """Beeswarm plot — global feature importance + hướng ảnh hưởng (4.2)."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_global_bar(shap_values, X_sample, save_path=None, max_display=15):
    """Bar chart mean(|SHAP value|) — dễ đọc cho báo cáo/slide hơn beeswarm."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def get_rejected_cases(model, X, threshold):
    """Lấy index các case bị model dự đoán Default=1 (bị 'từ chối') theo threshold đã chốt (0.176)."""
    proba = model.predict_proba(X)[:, 1]
    rejected_idx = np.where(proba >= threshold)[0]
    return rejected_idx, proba


def plot_local_waterfall(explainer, shap_values, X_sample, idx, save_path=None, max_display=10):
    """Waterfall plot — giải thích 1 case cụ thể (idx = vị trí trong X_sample), dùng cho
    các case bị 'từ chối' để trả lời 'AI nói Không — nhưng tại sao?'."""
    exp = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[idx].values,
        feature_names=X_sample.columns.tolist()
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def get_top_features_for_case(shap_values, X_sample, idx, top_n=5):
    """Top N feature ảnh hưởng nhiều nhất đến 1 case — input cho LLM integration (4.3)."""
    df = pd.DataFrame({
        'feature': X_sample.columns,
        'value': X_sample.iloc[idx].values,
        'shap_value': shap_values[idx]
    })
    df['abs_shap'] = df['shap_value'].abs()
    return df.sort_values('abs_shap', ascending=False).head(top_n)[['feature', 'value', 'shap_value']].reset_index(drop=True)
