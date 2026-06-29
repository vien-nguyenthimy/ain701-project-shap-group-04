import sys
from pathlib import Path
import os
from dotenv import load_dotenv

project_root = Path('d:/Python/AIN701_Group_04/demo').resolve()
sys.path.append(str(project_root))

from src.models.predict_model import predict_loan, load_model_and_features
from src.models.shap_analysis import get_explainer, get_top_features_for_case
from src.models.llm_explainer import explain_loan_decision

load_dotenv(project_root / ".env")

age = 19
income = 30000
loan_amount = 200000
credit_score = 450
months_employed = 3
num_credit_lines = 5
interest_rate = 22.5
loan_term = 60

r = (interest_rate / 100) / 12
n = loan_term
emi = loan_amount * r * ((1 + r)**n) / (((1 + r)**n) - 1)
existing_debt = 0.0
total_monthly_debt = existing_debt + emi
dti_ratio = (total_monthly_debt / (income / 12)) if income > 0 else 0.0

raw_input = {
    'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
    'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate, 
    'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': "Master's", 'EmploymentType': "Unemployed",
    'MaritalStatus': "Single", 'HasMortgage': "No", 'HasDependents': "No",
    'LoanPurpose': "Business", 'HasCoSigner': "No"
}

try:
    result, X_input = predict_loan(raw_input)
    model, feature_names = load_model_and_features()
    explainer = get_explainer(model)
    shap_values = explainer.shap_values(X_input)
    top_features = get_top_features_for_case(shap_values, X_input, idx=0, top_n=5)
    explanation = explain_loan_decision(result, top_features, raw_input)
    print("SUCCESS")
    print(explanation)
except Exception as e:
    import traceback
    traceback.print_exc()
