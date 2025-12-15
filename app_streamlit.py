import streamlit as st
import numpy as np
import joblib

# ===== إعدادات عامة =====
st.set_page_config(page_title="توقع القبول الجامعي", page_icon="🎓", layout="centered")

API_THRESHOLD = 70.0  # عتبة القبول %

st.title("🎓 نظام توقع القبول في الدراسات العليا")
st.caption("القرار: مقبول إذا كانت النسبة ≥ 70% وإلا مرفوض")

MODEL_PATH = "admission_model.joblib"
SCALER_PATH = "admission_scaler.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# ===== إدخالات المستخدم =====
col1, col2 = st.columns(2)

with col1:
    gre = st.number_input("درجة أختبار القبول (0-340)", min_value=0.0, max_value=340.0, value=320.0, step=1.0)
    rating = st.number_input("تصنيف الجامعة (1–5)", min_value=1, max_value=5, value=4, step=1)
    sop = st.number_input("قوة بيان الغرض (0-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.5)
    gpa = st.number_input("المعدل التراكمي (0-10)", min_value=0.0, max_value=10.0, value=9.0, step=0.01)

with col2:
    toefl = st.number_input("(120-0) TOEFL درجة أختبار", min_value=0.0, max_value=120.0, value=110.0, step=1.0)
    lor = st.number_input("قوة خطاب التوصية (0–5)", min_value=0.0, max_value=5.0, value=4.0, step=0.5)
    research = st.selectbox("خبرة بحثية", options=[0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

# ===== زر التنبؤ =====
if st.button("🔮 تنبؤ", use_container_width=True):
    x = np.array([[gre, toefl, rating, sop, lor, gpa, research]], dtype=float)
    x_s = scaler.transform(x)

    pred = float(model.predict(x_s)[0])
    pred = float(np.clip(pred, 0.0, 1.0))
    pct = pred * 100.0

    st.subheader("📊 النتيجة")
    st.metric("نسبة القبول", f"{pct:.2f}%")
    st.progress(int(round(pct)))

    if pct >= API_THRESHOLD:
        st.success(f"القرار النهائي: ✅ مقبول (العتبة: {API_THRESHOLD:.0f}%)")
    else:
        st.error(f"القرار النهائي: ❌ مرفوض (العتبة: {API_THRESHOLD:.0f}%)")

    with st.expander("عرض التفاصيل"):
        st.json({
            "احتمالية_القبول": pred,
            "نسبة_القبول_%": pct,
            "القرار_النهائي": "مقبول" if pct >= API_THRESHOLD else "مرفوض",
            "عتبة_القبول_%": API_THRESHOLD
        })
with st.expander(" كيف تم اتخاذ القرار؟"):
    st.markdown("""
    - يعتمد القرار على نموذج تعلم آلي
    - يتم تقييم:
        - GRE و TOEFL
        - المعدل التراكمي
        - قوة SOP و LOR
        - التصنيف الجامعي
        - الخبرة البحثية
    - العتبة المعتمدة: **70%**
    """)
