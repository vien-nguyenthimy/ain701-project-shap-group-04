import streamlit as st
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Đảm bảo import được src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.predict_model import predict_loan, load_model_and_features
from src.models.shap_analysis import get_explainer, get_top_features_for_case
from src.models.llm_explainer import explain_loan_decision

load_dotenv(project_root / ".env")

st.set_page_config(page_title="AI Loan Default Predictor", layout="wide")

st.title("🏦 Hệ thống Dự đoán & Giải thích Rủi ro Cho vay (AI)")

with st.sidebar:
    st.header("Nhập thông tin khách hàng")
    age = st.number_input("Tuổi (Age)", min_value=18, max_value=100, value=19)
    income = st.number_input("Thu nhập (USD) - Income", min_value=0, value=30000, step=1000)
    loan_amount = st.number_input("Số tiền vay (USD) - LoanAmount", min_value=0, value=200000, step=1000)
    credit_score = st.number_input("Điểm tín dụng (CreditScore)", min_value=300, max_value=850, value=450)
    months_employed = st.number_input("Số tháng làm việc (MonthsEmployed)", min_value=0, value=3)
    num_credit_lines = st.number_input("Số thẻ/khoản vay (NumCreditLines)", min_value=0, value=5)
    interest_rate = st.number_input("Lãi suất (%) - InterestRate", min_value=0.0, max_value=50.0, value=22.5, step=0.1)
    loan_term = st.number_input("Thời hạn vay (tháng) - LoanTerm", min_value=12, max_value=360, value=60, step=12)
    
    # Tính khoản trả góp hàng tháng (EMI) cho khoản vay mới
    r = (interest_rate / 100) / 12
    n = loan_term
    if r > 0 and n > 0:
        emi = loan_amount * r * ((1 + r)**n) / (((1 + r)**n) - 1)
    elif n > 0:
        emi = loan_amount / n
    else:
        emi = 0.0
        
    existing_debt = st.number_input("Dư nợ hiện tại hàng tháng (USD)", min_value=0.0, value=0.0, step=100.0)
    total_monthly_debt = existing_debt + emi
    dti_ratio = (total_monthly_debt / (income / 12)) if income > 0 else 0.0
    
    st.info(f"**Tỷ lệ Nợ/Thu nhập (DTI): {dti_ratio:.2f}**\n\n(Bao gồm trả góp khoản vay mới: ~{emi:,.0f} USD/tháng)")
    
    education = st.selectbox("Học vấn (Education)", ["High School", "Bachelor's", "Master's", "Ph.D."])
    employment = st.selectbox("Việc làm (EmploymentType)", ["Unemployed", "Part-time", "Full-time", "Self-employed"])
    marital = st.selectbox("Tình trạng hôn nhân (MaritalStatus)", ["Single", "Married", "Divorced"])
    mortgage = st.selectbox("Có vay thế chấp không? (HasMortgage)", ["No", "Yes"])
    dependents = st.selectbox("Có người phụ thuộc không? (HasDependents)", ["No", "Yes"])
    purpose = st.selectbox("Mục đích vay (LoanPurpose)", ["Business", "Education", "Home", "Auto", "Other"])
    cosigner = st.selectbox("Có người bảo lãnh không? (HasCoSigner)", ["No", "Yes"])
    
    predict_btn = st.button("🚀 Dự đoán", use_container_width=True)

if predict_btn:
    raw_input = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
        'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate, 
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education, 'EmploymentType': employment,
        'MaritalStatus': marital, 'HasMortgage': mortgage, 'HasDependents': dependents,
        'LoanPurpose': purpose, 'HasCoSigner': cosigner
    }
    
    with st.spinner("Đang phân tích dữ liệu..."):
        try:
            # 1. Dự đoán
            result, X_input = predict_loan(raw_input)
            
            # 2. Tính SHAP
            model, feature_names = load_model_and_features()
            explainer = get_explainer(model)
            shap_values = explainer.shap_values(X_input)
            top_features = get_top_features_for_case(shap_values, X_input, idx=0, top_n=5)
            
            # 3. LLM Giải thích
            explanation = explain_loan_decision(result, top_features)
            
            # Hiển thị kết quả
            st.subheader("📊 Kết quả Phân tích")
            
            col1, col2 = st.columns(2)
            with col1:
                status_color = "red" if result['prediction'] == 1 else "green"
                st.markdown(f"**Quyết định:** <span style='color:{status_color}; font-size:20px'>{result['status']}</span>", unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"**Xác suất vỡ nợ:** `{result['probability_default']:.1%}`")
            
            st.markdown("---")
            st.subheader("💡 Giải thích từ AI")
            st.info(explanation)
            
            with st.expander("🔍 Chi tiết Top 5 yếu tố ảnh hưởng (SHAP)"):
                st.dataframe(top_features)
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
            st.exception(e)
