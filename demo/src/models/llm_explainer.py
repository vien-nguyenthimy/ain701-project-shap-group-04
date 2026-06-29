"""
llm_explainer.py — Sinh giải thích tự nhiên từ kết quả SHAP (4.3)
"""
import os
from dotenv import load_dotenv
from google import genai
import pandas as pd

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_top_features(shap_values, X_input, feature_names, top_n=8):
    """Tạo bảng top features từ SHAP values"""
    vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    importance = pd.DataFrame({
        'feature': feature_names,
        'shap_value': vals,
        'value': X_input.iloc[0].values
    })
    importance['abs_shap'] = importance['shap_value'].abs()
    return importance.sort_values('abs_shap', ascending=False).head(top_n)


def build_explanation_prompt(result: dict, top_features) -> str:
    """Xây prompt từ kết quả predict_loan() + get_top_features_for_case()."""
    # Dịch tên cột sang tiếng Việt để LLM đọc hiểu tự nhiên hơn
    feature_translate = {
        'Age': 'Độ tuổi', 'Income': 'Thu nhập', 'LoanAmount': 'Số tiền vay',
        'CreditScore': 'Điểm tín dụng', 'MonthsEmployed': 'Thời gian làm việc (tháng)',
        'NumCreditLines': 'Số lượng thẻ/khoản vay', 'InterestRate': 'Lãi suất',
        'LoanTerm': 'Thời hạn vay (tháng)', 'DTIRatio': 'Tỷ lệ Nợ/Thu nhập',
        'Education': 'Trình độ học vấn', 'EmploymentType': 'Tình trạng việc làm',
        'MaritalStatus': 'Tình trạng hôn nhân', 'HasMortgage': 'Có vay thế chấp',
        'HasDependents': 'Có người phụ thuộc', 'LoanPurpose': 'Mục đích vay',
        'HasCoSigner': 'Có người bảo lãnh'
    }

    feature_lines = []
    for _, row in top_features.iterrows():
        direction = "tăng" if row['shap_value'] > 0 else "giảm"
        feat_vn = feature_translate.get(row['feature'], row['feature'])
        feature_lines.append(f"- {feat_vn}: {row['value']} (yếu tố làm {direction} rủi ro)")
    feature_text = "\n".join(feature_lines)

    status_vn = "Chưa đủ điều kiện duyệt (Rủi ro cao)" if result['prediction'] == 1 else "Đủ điều kiện duyệt (An toàn)"

    return f"""Bạn là một Chuyên viên Phê duyệt Tín dụng (Credit Analyst) cấp cao tại ngân hàng, với phong cách làm việc vô cùng chuyên nghiệp, minh bạch nhưng cũng rất tinh tế và thấu hiểu khách hàng.
Hệ thống vừa đánh giá hồ sơ vay của khách hàng với kết quả:

- Quyết định: {status_vn}
- Tỷ lệ rủi ro hệ thống đánh giá: {result['probability_default']:.1%}

Các yếu tố chính dẫn đến quyết định này (dựa trên phân tích dữ liệu):
{feature_text}

Nhiệm vụ của bạn:
Hãy viết một thư phản hồi gửi trực tiếp cho khách hàng.
Yêu cầu ĐẶC BIỆT:
- KHÔNG VIẾT ĐOẠN VĂN DÀI DÒNG. Trình bày cực kỳ **ngắn gọn, súc tích**.
- BẮT BUỘC sử dụng chữ in đậm (**bold**) và gạch đầu dòng (-) để highlight (làm nổi bật) các thông số quan trọng, lý do và lời khuyên.

Yêu cầu về cấu trúc (Thật ngắn gọn):
1. Lời chào & Kết quả (1 câu lịch sự).
2. Tại sao lại có quyết định này? (2-3 gạch đầu dòng): Giải thích ngắn gọn lý do, nhớ in đậm các con số quan trọng (VD: **19 tuổi**, **thu nhập 30.000 USD**, **lãi suất 22.5%**).
3. Cách cải thiện hồ sơ (2-3 gạch đầu dòng): Đưa ra các giải pháp khắc phục cực kỳ thiết thực, in đậm từ khóa chính (VD: **Giảm số tiền vay**, **Đợi công việc ổn định hơn**).

Lưu ý:
- Giọng văn: Chuyên nghiệp, thấu hiểu (xưng hô "chúng tôi" và "Quý khách/Anh/Chị").
- Tuyệt đối KHÔNG dùng các từ ngữ kỹ thuật như "AI", "SHAP", "Model", "Log-odds", "Feature", "Tăng/giảm rủi ro"."""


def generate_explanation(prompt: str, model_name: str = "gemini-3.5-flash") -> str:
    """Gọi Gemini API, trả về đoạn giải thích."""
    response = client.models.generate_content(
        model=model_name, 
        contents=prompt,
        config={'temperature': 0.6}  # Tăng tính tự nhiên, sáng tạo cho câu văn
    )
    return response.text

def explain_loan_decision(result: dict, top_features) -> str:
    """Hàm tổng — dùng trong app.py: result, top_features có sẵn từ predict_model.py + shap_analysis.py"""
    prompt = build_explanation_prompt(result, top_features)
    return generate_explanation(prompt)