import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS - Dark Theme
# ============================================
st.markdown("""
<style>
    .stApp {
        # background-color: #0e1117;
        color: #e0e0e0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #c05660 0%, #a04550 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(192, 86, 96, 0.3);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1.05rem;
    }
    
    .input-card {
        background: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        border-left: 4px solid #c05660;
    }
    
    .result-safe {
        background: linear-gradient(135deg, #1e5638 0%, #164232 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: #86efac;
        box-shadow: 0 4px 15px rgba(30, 86, 56, 0.4);
        margin-top: 1.5rem;
        border: 1px solid #22c55e;
    }
    .result-safe h2 {
        margin: 0;
        color: #86efac;
        font-size: 2rem;
    }
    
    .result-danger {
        background: linear-gradient(135deg, #7f1d1d 0%, #5c1515 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: #fca5a5;
        box-shadow: 0 4px 15px rgba(127, 29, 29, 0.4);
        margin-top: 1.5rem;
        border: 1px solid #ef4444;
    }
    .result-danger h2 {
        margin: 0;
        color: #fca5a5;
        font-size: 2rem;
    }
    
    [data-testid=stSidebar] {
        background-color: #1e293b !important;
    }
    [data-testid=stSidebar] * {
        color: #e2e8f0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #c05660 0%, #a04550 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        width: 100%;
        box-shadow: 0 2px 8px rgba(192, 86, 96, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(192, 86, 96, 0.5);
    }
    
    /* ปรับ input fields */
    .stNumberInput > div > div > input {
        background-color: #1e293b;
        color: #e0e0e0;
        border: 1px solid #334155;
    }
    
    /* ปรับ selectbox */
    .stSelectbox > div > div > select {
        background-color: #1e293b;
        color: #e0e0e0;
    }
    
    /* ปรับ metric cards */
    [data-testid="stMetricValue"] {
        color: #c05660 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# โหลดโมเดล
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์โมเดล กรุณารัน train_model.py ก่อน")
    st.stop()

# ============================================
# Header
# ============================================
st.markdown("""
<div class="main-header">
    <h1>🫀 Heart Disease Predictor</h1>
    <p>ระบบประเมินความเสี่ยงโรคหัวใจด้วย Decision Tree AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.markdown("### 📊 ข้อมูลโมเดล")
    st.markdown("""
    **Algorithm:** Decision Tree Classifier  
    **Dataset:** Heart Disease (920 samples)  
    **Features:** 11 clinical features  
    **Target:** Binary (Disease / No Disease)
    """)
    
    st.markdown("---")
    st.markdown("### 🩺 คำอธิบาย Features")
    st.markdown("""
    - **Age** - อายุ (ปี)
    - **Sex** - เพศ (0=หญิง, 1=ชาย)
    - **ChestPainType** - อาการเจ็บหน้าอก
      - 1: ATA, 2: NAP, 3: ASY, 4: TA
    - **RestingBP** - ความดันโลหิต (mmHg)
    - **Cholesterol** - โคเลสเตอรอล (mg/dl)
    - **FastingBS** - น้ำตาลตอนอดอาหาร (>120=1)
    - **RestingECG** - ผล ECG (0-2)
    - **MaxHR** - อัตราการเต้นหัวใจสูงสุด
    - **ExerciseAngina** - เจ็บหน้าอกขณะออกกำลังกาย
    - **Oldpeak** - ST depression
    - **ST_Slope** - ความชัน ST segment (1-3)
    """)
    
    st.markdown("---")
    st.info("️ แอปนี้เป็นเพียงเครื่องมือช่วยประเมินเบื้องต้น")

# ============================================
# Input Section
# ============================================
st.markdown("### 📝 กรอกข้อมูลสุขภาพ")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    age = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=55)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    sex = st.selectbox("⚧ เพศ", options=[(1, "ชาย"), (0, "หญิง")], format_func=lambda x: x[1])
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    chest_pain = st.selectbox("💔 อาการเจ็บหน้าอก",
        options=[(1, "ATA"), (2, "NAP"), (3, "ASY"), (4, "TA")],
        format_func=lambda x: x[1])
    st.markdown("</div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    resting_bp = st.number_input(" ความดันโลหิต", min_value=80, max_value=200, value=130)
    st.markdown("</div>", unsafe_allow_html=True)
with col5:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    cholesterol = st.number_input("🧪 โคเลสเตอรอล", min_value=0, max_value=600, value=220)
    st.markdown("</div>", unsafe_allow_html=True)
with col6:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    fasting_bs = st.selectbox("🍬 น้ำตาล>120",
        options=[(0, "ไม่"), (1, "ใช่")], format_func=lambda x: x[1])
    st.markdown("</div>", unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)
with col7:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    resting_ecg = st.selectbox(" ECG",
        options=[(0, "Normal"), (1, "Abnormal"), (2, "Hypertrophy")],
        format_func=lambda x: x[1])
    st.markdown("</div>", unsafe_allow_html=True)
with col8:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    max_hr = st.number_input("💓 Max HR", min_value=60, max_value=220, value=140)
    st.markdown("</div>", unsafe_allow_html=True)
with col9:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    exercise_angina = st.selectbox("🏃 เจ็บหน้าอก",
        options=[(0, "ไม่"), (1, "ใช่")], format_func=lambda x: x[1])
    st.markdown("</div>", unsafe_allow_html=True)

col10, col11 = st.columns(2)
with col10:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    oldpeak = st.number_input("📉 Oldpeak", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)
with col11:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st_slope = st.selectbox("📊 ST Slope",
        options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")],
        format_func=lambda x: x[1])
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Prediction
# ============================================
if st.button("🔮 ประเมินความเสี่ยง"):
    
    input_data = np.array([[
        age, sex[0], chest_pain[0], resting_bp, cholesterol,
        fasting_bs[0], resting_ecg[0], max_hr, exercise_angina[0], oldpeak, st_slope[0]
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    risk_prob = probabilities[1] * 100
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-danger">
            <h2>⚠️ พบความเสี่ยงโรคหัวใจ</h2>
            <p style="font-size: 1.2rem;">ระดับความเสี่ยง: <strong>{risk_prob:.1f}%</strong></p>
            <p>แนะนำให้ปรึกษาแพทย์</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-safe">
            <h2>✅ ไม่พบความเสี่ยง</h2>
            <p style="font-size: 1.2rem;">ระดับความเสี่ยง: <strong>{risk_prob:.1f}%</strong></p>
            <p>รักษาสุขภาพต่อไป</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chart
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
    
    fig.add_trace(go.Pie(
        labels=['ไม่เสี่ยง', 'เสี่ยง'],
        values=[probabilities[0]*100, probabilities[1]*100],
        marker_colors=['#22c55e', '#ef4444'],
        hole=0.4,
        textfont=dict(color='white')
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=['ไม่เสี่ยง', 'เสี่ยง'],
        y=[probabilities[0]*100, probabilities[1]*100],
        marker_color=['#22c55e', '#ef4444']
    ), row=1, col=2)
    
    fig.update_layout(
        height=350, 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0')
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; padding: 1rem;'>🫀 Heart Disease Predictor © 2026</div>", unsafe_allow_html=True)