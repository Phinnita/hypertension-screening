import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Advanced AI Hypertension Screen", page_icon="🟢", layout="centered")

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        padding: 0.6rem 2rem;
    }
    div.stButton > button:first-child:hover { background-color: #1b5e20; }
    hr { border-top: 2px solid #c8e6c9; }
    h1 { color: #1b5e20; }
    h2, h3 { color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    data = joblib.load("hypertension_svm_system.pkl")
    return data["model"], data["scaler"]

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ 'hypertension_svm_system.pkl' กรุณาตรวจสอบการใส่ไฟล์ในโฟลเดอร์ให้ถูกต้อง")
    st.stop()

st.title("🟢 ระบบคัดกรองความเสี่ยงโรคความดันโลหิตสูงขั้นสูง")
st.write("วิเคราะห์สภาวะความดันโลหิตร่วมกับปัจจัยทางพฤติกรรมและการนอนหลับด้วยปัญญาประดิษฐ์ (AI SVM)")
st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนที่ 1: ข้อมูลทั่วไป
st.subheader("👤 1. ข้อมูลทั่วไปและสรีระวิทยา")
col_a, col_b = st.columns(2)
with col_a:
    gender = st.radio("เพศสภาพ (Gender)", ["หญิง (Female)", "ชาย (Male)"])
    age = st.slider("อายุ (ปี)", min_value=1, max_value=120, value=40)
with col_b:
    weight = st.number_input("น้ำหนักตัว (กิโลกรัม)", min_value=10.0, max_value=250.0, value=65.0)
    height = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=100.0, max_value=250.0, value=165.0)

bmi = weight / ((height / 100.0) ** 2)
st.info(f"📊 ดัชนีมวลกาย (BMI) ของคุณคือ: **{bmi:.2f}**")
st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนที่ 2: พฤติกรรมและการนอน (ออกกำลังกาย + นอนกรน)
st.subheader("🏃‍♂️ 2. พฤติกรรมการใช้ชีวิตและคุณภาพการนอนหลับ")
col_c, col_d = st.columns(2)
with col_c:
    family_history = st.selectbox("มีญาติสายตรงเป็นโรคความดันโลหิตสูงหรือไม่?", ["ไม่มีประวัติในครอบครัว", "มีพ่อหรือแม่เป็น", "ทั้งพ่อและแม่เป็น"])
    pedigree = 0.20 if family_history == "ไม่มีประวัติในครอบครัว" else (0.50 if family_history == "มีพ่อหรือแม่เป็น" else 0.80)
    
    exercise_habit = st.selectbox("ความถี่ในการออกกำลังกาย (อย่างน้อย 30 นาที/ครั้ง)", ["แทบไม่ได้ออกกำลังกายเลย (Sedentary)", "ออกกำลังกายบ้าง (1-2 วัน/สัปดาห์)", "ออกกำลังกายสม่ำเสมอ (3 วันขึ้นไป/สัปดาห์)"])
    exercise = 0 if exercise_habit == "แทบไม่ได้ออกกำลังกายเลย (Sedentary)" else (1 if exercise_habit == "ออกกำลังกายบ้าง (1-2 วัน/สัปดาห์)" else 2)
with col_d:
    sleep_hours = st.slider("จำนวนชั่วโมงการนอนหลับเฉลี่ยต่อคืน", min_value=3, max_value=12, value=7)
    sleep_apnea_check = st.checkbox("📢 มีอาการนอนกรนเสียงดังสะดุด มีเสียงครืดคราด หรือสะดุ้งตื่นมาสำลักน้ำลายกลางดึก")
    sleep_apnea = 1 if sleep_apnea_check else 0

st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนที่ 3: สัญญาณชีพ
st.subheader("🩺 3. พฤติกรรมการกินและสัญญาณชีพ")
col1, col2 = st.columns(2)
with col1:
    pct_sodium = st.slider("สัดส่วนการกินอาหารเค็ม / แปรรูป / โซเดียมสูง (%)", min_value=0, max_value=100, value=35)
    sbp = st.number_input("ความดันโลหิตตัวบน (Systolic BP) [mm Hg]", min_value=70, max_value=230, value=120)
with col2:
    heart_rate = st.number_input("อัตราการเต้นของหัวใจขณะพัก [ครั้ง/นาที]", min_value=40, max_value=180, value=75)
    dbp = st.number_input("ความดันโลหิตตัวล่าง (Diastolic BP) [mm Hg]", min_value=40, max_value=140, value=80)

st.markdown("<hr>", unsafe_allow_html=True)

# ประมวลผลลัพธ์
btn_1, btn_2 = st.columns([4, 1])
with btn_1:
    submit = st.button("🔮 เริ่มวิเคราะห์ความเสี่ยงองค์รวมเชิงลึก", use_container_width=True)
with btn_2:
    if st.button("🔄 ล้างข้อมูล", use_container_width=True): st.rerun()

if submit:
    input_data = np.array([[sbp, dbp, heart_rate, bmi, pedigree, age, sleep_hours, pct_sodium, exercise, sleep_apnea]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("### 📝 รายงานผลการประเมินสุขภาพ:")
    if prediction == 1 or sbp >= 140 or dbp >= 90:
        st.error("🔴 **ผลลัพธ์จาก AI: มีความเสี่ยงสูงต่อภาวะโรคความดันโลหิตสูง**")
        st.markdown("### ⚠️ ปัจจัยเสี่ยงวิกฤตที่ต้องเฝ้าระวังของคุณ:")
        if sleep_apnea == 1:
            st.warning("• **ภาวะสงสัยการหยุดหายใจขณะหลับ (Obstructive Sleep Apnea):** การนอนกรนสะดุดทำให้ออกซิเจนในเลือดลดลงต่ำในตอนกลางคืน กระตุ้นความดันโลหิตให้พุ่งสูงเรื้อรัง แนะนำให้พบแพทย์เพื่อทำ Sleep Test")
        if exercise == 0:
            st.warning("• **พฤติกรรมเนือยนิ่ง / ขาดการออกกำลังกาย:** ส่งผลให้ความยืดหยุ่นของผนังหลอดเลือดลดลง แนะนำให้เดินเร็วสะสมให้ได้ 150 นาทีต่อสัปดาห์")
    else:
        st.success("🟢 **ผลลัพธ์จาก AI: สัญญาณชีพและพฤติกรรมอยู่ในเกณฑ์เสี่ยงต่ำ (ปกติ)**")
        st.info("🎉 ข้อแนะนำ: ผลวิเคราะห์อยู่ในเกณฑ์ดีมาก! รักษาวินัยการนอนหลับและการขยับร่างกายที่ดีแบบนี้ต่อไปยาวๆ ครับ")

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.caption("ℹ️ **Medical Disclaimer:** ระบบนี้เป็นเพียงการคัดกรองความเสี่ยงเบื้องต้นด้วยแบบจำลองคณิตศาสตร์ AI เท่านั้น ไม่สามารถใช้ทดแทนการวินิจฉัยโรคโดยแพทย์ในสถานพยาบาลได้")