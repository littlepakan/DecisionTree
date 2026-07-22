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
    page_title="🫀 Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS - Cardiac Theme
# ============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe4e4 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 50%, #8e2323 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 30px rgba(231, 76, 60, 0.4);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        position: relative;
        z-index: 1;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    .input-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(231, 76, 60, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #e74c3c;
    }
    
    .result-safe {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fdbac 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 30px rgba(127, 219, 172, 0.4);
        margin-top: 1.5rem;
    }
    
    .result-danger {
        background: linear-gradient(135deg, #ff6b6b 0%, #c0392b 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 30px rgba(192, 57, 43, 0.4);
        margin-top: 1.5rem;
    }
    
    .result-card h2 {
        margin: 0;
        color: white;
        font-size: 2.2rem;
    }
    .result-card p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    [data-testid=stSidebar] {
        background: linear-gradient(180deg, #2c1810 0%, #4a1f1f 100%);
        color: white;
    }
    [data-testid=stSidebar] h1, [data-testid=stSidebar] h2, [data-testid=stSidebar] h3 {
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.5);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
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
    st.error("❌ ไม่พบไฟล์โมเดล กรุณาวางไฟล์ในโฟลเดอร์เดียวกับ app.py")
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
    - **ChestPainType** - ลักษณะอาการเจ็บหน้าอก
      - 1: ATA (Angina)
      - 2: NAP (Non-Anginal)
      - 3: ASY (Asymptomatic)
      - 4: TA (Typical Angina)
    - **RestingBP** - ความดันโลหิตขณะพัก (mmHg)
    - **Cholesterol** - โคเลสเตอรอล (mg/dl)
    - **FastingBS** - น้ำตาลในเลือดตอนอดอาหาร (>120 = 1)
    - **RestingECG** - ผล ECG ขณะพัก (0-2)
    - **MaxHR** - อัตราการเต้นหัวใจสูงสุด
    - **ExerciseAngina** - เจ็บหน้าอกขณะออกกำลังกาย
    - **Oldpeak** - ST depression
    - **ST_Slope** - ความชันของ ST segment (1-3)
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ คำเตือน")
    st.info("แอปนี้เป็นเพียงเครื่องมือช่วยประเมินเบื้องต้น ไม่สามารถทดแทนการวินิจฉัยจากแพทย์ได้")

# ============================================
# Input Section
# ============================================
st.markdown("### 📝 กรอกข้อมูลสุขภาพของผู้ป่วย")

# Row 1: Demographics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    age = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=55, step=1)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    sex = st.selectbox("⚧ เพศ", options=[(1, "ชาย"), (0, "หญิง")], format_func=lambda x: x[1])
    sex_val = sex[0]
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    chest_pain = st.selectbox(
        "💔 ลักษณะอาการเจ็บหน้าอก",
        options=[
            (1, "ATA - Angina"),
            (2, "NAP - Non-Anginal"),
            (3, "ASY - Asymptomatic"),
            (4, "TA - Typical Angina")
        ],
        format_func=lambda x: x[1]
    )
    chest_pain_val = chest_pain[0]
    st.markdown("</div>", unsafe_allow_html=True)

# Row 2: Vital Signs
col4, col5, col6 = st.columns(3)
with col4:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    resting_bp = st.number_input("🩸 ความดันโลหิตขณะพัก (mmHg)", 
                                  min_value=80, max_value=200, value=130, step=1)
    st.markdown("</div>", unsafe_allow_html=True)
with col5:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    cholesterol = st.number_input("🧪 โคเลสเตอรอล (mg/dl)", 
                                   min_value=0, max_value=600, value=220, step=1)
    st.markdown("</div>", unsafe_allow_html=True)
with col6:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    fasting_bs = st.selectbox("🍬 น้ำตาลในเลือดตอนอดอาหาร > 120?",
                               options=[(0, "ไม่ (≤ 120)"), (1, "ใช่ (> 120)")],
                               format_func=lambda x: x[1])
    fasting_bs_val = fasting_bs[0]
    st.markdown("</div>", unsafe_allow_html=True)

# Row 3: ECG & Exercise
col7, col8, col9 = st.columns(3)
with col7:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    resting_ecg = st.selectbox("📈 ผล ECG ขณะพัก",
                                options=[(0, "0 - Normal"), 
                                         (1, "1 - ST-T wave abnormality"),
                                         (2, "2 - LV hypertrophy")],
                                format_func=lambda x: x[1])
    ecg_val = resting_ecg[0]
    st.markdown("</div>", unsafe_allow_html=True)
with col8:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    max_hr = st.number_input("💓 อัตราการเต้นหัวใจสูงสุด (bpm)",
                              min_value=60, max_value=220, value=140, step=1)
    st.markdown("</div>", unsafe_allow_html=True)
with col9:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    exercise_angina = st.selectbox("🏃 เจ็บหน้าอกขณะออกกำลังกาย?",
                                    options=[(0, "ไม่"), (1, "ใช่")],
                                    format_func=lambda x: x[1])
    ex_angina_val = exercise_angina[0]
    st.markdown("</div>", unsafe_allow_html=True)

# Row 4: ST Segment
col10, col11 = st.columns([1, 1])
with col10:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    oldpeak = st.number_input("📉 ST Depression (Oldpeak)",
                               min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
                               format="%.1f")
    st.markdown("</div>", unsafe_allow_html=True)
with col11:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st_slope = st.selectbox("📊 ความชันของ ST Segment",
                             options=[(1, "1 - Upsloping"),
                                      (2, "2 - Flat"),
                                      (3, "3 - Downsloping")],
                             format_func=lambda x: x[1])
    st_slope_val = st_slope[0]
    st.markdown("</div>", unsafe_allow_html=True)

# ปุ่มทำนาย
st.markdown("")
if st.button("🔮 ประเมินความเสี่ยงโรคหัวใจ", use_container_width=True):
    
    # เตรียมข้อมูล
    input_data = np.array([[
        age, sex_val, chest_pain_val, resting_bp, cholesterol,
        fasting_bs_val, ecg_val, max_hr, ex_angina_val, oldpeak, st_slope_val
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    # ทำนาย
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    risk_prob = probabilities[1] * 100
    
    # แสดงผลตามความเสี่ยง
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card result-danger">
            <h2>⚠️ พบความเสี่ยงโรคหัวใจ</h2>
            <p style="font-size: 1.3rem; margin-top: 0.5rem;">
                ระดับความเสี่ยง: <strong>{risk_prob:.1f}%</strong>
            </p>
            <p style="margin-top: 1rem; opacity: 0.95;">
                แนะนำให้ปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <h2>✅ ไม่พบความเสี่ยงโรคหัวใจ</h2>
            <p style="font-size: 1.3rem; margin-top: 0.5rem;">
                ระดับความเสี่ยง: <strong>{risk_prob:.1f}%</strong>
            </p>
            <p style="margin-top: 1rem; opacity: 0.95;">
                รักษาสุขภาพต่อไปด้วยการออกกำลังกายและอาหารที่มีประโยชน์
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability Chart
    st.markdown("### 📊 การกระจายความเสี่ยง")
    
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=("สัดส่วนความเสี่ยง", "เปรียบเทียบ"))
    
    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=['ไม่เสี่ยง', 'เสี่ยง'],
            values=[probabilities[0]*100, probabilities[1]*100],
            marker_colors=['#27ae60', '#e74c3c'],
            hole=0.4,
            textinfo='percent+label'
        ),
        row=1, col=1
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=['ไม่เสี่ยง', 'เสี่ยง'],
            y=[probabilities[0]*100, probabilities[1]*100],
            marker_color=['#27ae60', '#e74c3c'],
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis2=dict(range=[0, 110])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # แสดงข้อมูล input
    with st.expander("📋 ดูข้อมูลที่ใช้ทำนาย"):
        df_input = pd.DataFrame({
            'Feature': ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                        'Oldpeak', 'ST_Slope'],
            'Value': [age, sex[1], chest_pain[1], resting_bp, cholesterol,
                      fasting_bs[1], resting_ecg[1], max_hr, exercise_angina[1],
                      oldpeak, st_slope[1]]
        })
        st.dataframe(df_input, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>"
    "🫀 Built with <strong>Decision Tree</strong> & <strong>Streamlit</strong> | "
    "Heart Disease Prediction System © 2026"
    "</div>", 
    unsafe_allow_html=True
)