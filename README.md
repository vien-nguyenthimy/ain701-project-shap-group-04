# **Xây dựng hệ thống chấm điểm tín dụng minh bạch: Kết hợp TreeSHAP và LLM trong giải thích quyết định vay vốn**
# AI Nói Không — Nhưng Tại Sao?
### Explainable AI cho Credit Scoring — AIN701, Group 04

Hệ thống dự đoán rủi ro vỡ nợ khoản vay (loan default), kết hợp **XGBoost** với **SHAP (TreeSHAP)** để giải thích quyết định, và **LLM (Gemini API)** để chuyển giải thích kỹ thuật thành ngôn ngữ tự nhiên — đóng gói trong demo chatbox bằng Streamlit.

---

## 1. Mục tiêu

Các mô hình ML cho vay thường chỉ trả về 1 con số (xác suất default) mà không nói được **lý do**. Project này giải quyết đúng vấn đề đó: với mỗi hồ sơ vay, hệ thống không chỉ đưa ra quyết định Approve/Reject, mà còn giải thích **vì sao** bằng cả biểu đồ (SHAP) và văn bản tự nhiên (LLM).

## 2. Kiến trúc pipeline

```
Hồ sơ vay (form/chatbox)
        │
        ▼
Preprocessing & Predict  →  Encode input + threshold đã tune → Quyết định
        │
        ▼
SHAP Explainer (TreeSHAP)
        ├── Global Chart  → feature importance tổng thể (báo cáo, kiểm tra model học đúng logic)
        └── Local Chart   → top features cho đúng case này
                │
                ▼
        LLM (Gemini API) → sinh giải thích tự nhiên từ top features
                │
                ▼
        Chatbox hiển thị: Quyết định + Chart + Giải thích
```

## 3. Cấu trúc thư mục

```
demo/
├── data/
│   ├── raw/Loan_default.csv          # Dataset gốc (255,347 dòng, 18 cột)
│   ├── processed/                    # X_train/X_test/y_train/y_test, feature_names.txt
│   └── models/                       # xgboost.pkl, lightgbm.pkl, catboost.pkl
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_SHAP_Analysis.ipynb
│   └── 04_LLM_Explainer.ipynb
├── src/
│   ├── data/        load_data.py, processing.py
│   ├── models/      train_model.py, predict_model.py, shap_analysis.py, llm_explainer.py
│   └── reports/figures/
├── app/app.py                        # Streamlit chatbox demo
├── main.py                           # Chạy toàn bộ pipeline (load → train → save) bằng 1 lệnh
├── requirements.txt
└── .env.example                      # Mẫu khai báo GOOGLE_API_KEY (không chứa key thật)
```

## 4. Cài đặt

```bash
git clone https://github.com/vien-nguyenthimy/ain701-project-shap-group-04.git
cd ain701-project-shap-group-04/demo
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Tạo file `.env` (copy từ `.env.example`), điền API key Gemini của riêng bạn:
```
GOOGLE_API_KEY=your_api_key_here
```
Lấy key free tại https://aistudio.google.com/apikey. **Không commit file `.env`** — đã có trong `.gitignore`.

## 5. Cách chạy

**Tái lập toàn bộ pipeline (data → train → save model):**
```bash
python main.py
```

**Chạy từng bước chi tiết (khuyến nghị để xem rõ phân tích):**
1. `01_EDA_Preprocessing.ipynb` — khám phá + xử lý dữ liệu, lưu `X_train/X_test`
2. `02_Model_Training.ipynb` — train + so sánh XGBoost/LightGBM/CatBoost
3. `03_SHAP_Analysis.ipynb` — global + local explanation bằng TreeSHAP
4. `04_LLM_Explainer.ipynb` — sinh giải thích tự nhiên từ top features SHAP

**Chạy demo chatbox:**
```bash
streamlit run demo/app/app.py
```


## 6. Thành viên nhóm — AIN701 Group 04

| Họ tên | Vai trò |
|---|---|
| Nguyễn Thị Mỹ Viên | SHAP Analysis (4.2), LLM Integration (4.3) |
| *Lê Thị Bích Trà* | *EDA, Processing (3.2, 3.3)* |
| *Văn Thị Tuyết Nga* | *Demo Streamlit (4.4)* |
| *Võ Hiếu Nghĩa* | *Traing model(4.1)* |
| *Lê Hoàng An* | *Cơ sở lý thuyết (2.1, 2.2, 2.3)* |
| *Võ Quang Huy* | *Cơ sở lý thuyết (2.4, 2.5, 2.6)* |
| *Lê Nguyễn Như Quỳnh* | *Chương 1, Slide* |
| *Nguyễn Thị Thùy Linh* | *Chương 5, Slide* |

## 7. Tech stack

`Python` · `XGBoost` · `LightGBM` ·  `scikit-learn` · `SHAP` · `imbalanced-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib` · `Google Gemini API` (`google-genai`) · `python-dotenv` · `Streamlit`
