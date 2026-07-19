import streamlit as st 
import numpy as np
import joblib

# 🛠 [จุดที่แก้ไข] ปรับเป็นภาษาไทยเพื่อให้เวลาแชร์ในแชทแล้วขึ้นชื่อระบบสวยๆ เหมือนของเพื่อนครับ
st.set_page_config(
    page_title="ระบบทำนายความเสี่ยงโรคความดันโลหิตสูง", 
    page_icon="🩺", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# ปรับแต่ง CSS ผสานความโมเดิร์นและการจัดรูปแบบ UI ให้นุ่มนวลขึ้น
st.markdown("""
    <style>
    /* ปรับแต่งปุ่มกดหลัก */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        padding: 0.75rem 2rem;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1b5e20 0%, #0d3c12 100%);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* ปรับแต่งปุ่มล้างข้อมูล */
    div.stButton > button:secondary {
        border-radius: 12px;
        padding: 0.75rem 2rem;
    }
    
    /* ปรับแต่งเส้นคั่นและการ์ดกลุ่มข้อมูล */
    hr { border-top: 2px solid #e0e0e0; margin-top: 15px; margin-bottom: 25px; }
    h1 { color: #1b5e20; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    h2, h3 { color: #2e7d32; font-family: 'Helvetica Neue', sans-serif; font-weight: 650; }
    
    /* ปรับแต่งสไตล์ของ Container ข้อมูล */
    .block-container { padding-top: 2rem; }
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

# ส่วนหัวของแอปพลิเคชัน
st.title("🩺 ระบบคัดกรองความเสี่ยงโรคความดันโลหิตสูงด้วย AI")
st.write("เครื่องมือวิเคราะห์ทางสถิติและไลฟ์สไตล์ด้วยปัญญาประดิษฐ์ชั้นสูง (Advanced SVM Machine Learning Model)")
st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนที่ 1: ข้อมูลทั่วไปและดัชนีมวลกาย
with st.container():
    st.subheader("👤 1. ข้อมูลพื้นฐานทางกายภาพ")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        gender = st.radio("เพศสภาพตามกำเนิด (Gender)", ["หญิง (Female)", "ชาย (Male)"])
        age = st.slider("อายุในปัจจุบัน (ปี)", min_value=1, max_value=120, value=40)
    with col_b:
        weight = st.number_input("น้ำหนักตัว (กิโลกรัม)", min_value=10.0, max_value=250.0, value=65.0, step=0.1)
        height = st.number_input("ส่วนสูง (เซนติเมตร)", min_value=100.0, max_value=250.0, value=165.0, step=0.1)

    bmi = weight / ((height / 100.0) ** 2)
    
    c1, c2 = st.columns([1, 2])
    c1.metric(label="ดัชนีมวลกาย (BMI)", value=f"{bmi:.2f}")
    if bmi < 18.5:
        c2.caption("ℹ️ สถานะ: น้ำหนักน้อยกว่าเกณฑ์ (Underweight)")
    elif 18.5 <= bmi < 23:
        c2.caption("ℹ️ สถานะ: น้ำหนักปกติ เหมาะสม (Healthy Weight)")
    elif 23 <= bmi < 25:
        c2.caption("ℹ️ สถานะ: น้ำหนักเกินมาตรฐาน (Overweight)")
    else:
        c2.caption("ℹ️ ภาวะน้ำหนักเกินเกณฑ์ / โรคอ้วน (Obesity)")
        
st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนที่ 2: ไลฟ์สไตล์และคุณภาพการนอนหลับ
with st.container():
    st.subheader("🏃‍♂️ 2. พฤติกรรมการใช้ชีวิตและคุณภาพการนอนหลับ")
    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        family_history = st.selectbox(
            "ประวัติการเป็นโรคความดันโลหิตสูงในครอบครัว (สายตรง)", 
            ["ไม่มีประวัติในครอบครัว", "มีพ่อหรือแม่เป็น", "ทั้งพ่อและแม่เป็น"]
        )
        pedigree = 0.20 if family_history == "ไม่มีประวัติในครอบครัว" else (0.50 if family_history == "มีพ่อหรือแม่เป็น" else 0.80)
        
        exercise_habit = st.selectbox(
            "ความถี่ในการออกกำลังกาย (ต่อเนื่อง 30 นาทีขึ้นไป)", 
            ["แทบไม่ได้ออกกำลังกายเลย (Sedentary)", "ออกกำลังกายบ้าง (1-2 วัน/สัปดาห์)", "ออกกำลังกายสม่ำเสมอ (3 วันขึ้นไป/สัปดาห์)"]
        )
        exercise = 0 if exercise_habit == "แทบไม่ได้ออกกำลังกายเลย (Sedentary)" else (1 if exercise_habit == "ออกกำลังกายบ้าง (1-2 วัน/สัปดาห์)" else 2)
    with col_d:
        sleep_hours = st.slider("จำนวนชั่วโมงการนอนหลับเฉลี่ยต่อคืน", min_value=3, max_value=12, value=7)
        st.write("") 
        sleep_apnea_check = st.checkbox("📢 พบอาการนอนกรนเสียงดังสะดุด ครืดคราด หรือสะดุ้งตื่นมาสำลักกลางดึก")
        sleep_apnea = 1 if sleep_apnea_check else 0

st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนที่ 3: สัญญาณชีพและพฤติกรรมการบริโภค
with st.container():
    st.subheader("🩺 3. สัญญาณชีพเชิงลึกทางการแพทย์")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        pct_sodium = st.slider("สัดส่วนพฤติกรรมการบริโภคโซเดียม/อาหารแปรรูปประจำวัน (%)", min_value=0, max_value=100, value=35)
        sbp = st.number_input("ความดันโลหิตตัวบน (Systolic BP) [mm Hg]", min_value=70, max_value=230, value=120, step=1)
    with col2:
        heart_rate = st.number_input("อัตราการเต้นของหัวใจขณะพัก (Resting Heart Rate) [ครั้ง/นาที]", min_value=40, max_value=180, value=75, step=1)
        dbp = st.number_input("ความดันโลหิตตัวล่าง (Diastolic BP) [mm Hg]", min_value=40, max_value=140, value=80, step=1)

st.markdown("<hr>", unsafe_allow_html=True)

# การจัดวางปุ่มประมวลผล
btn_1, btn_2 = st.columns([4, 1])
with btn_1:
    submit = st.button(" เริ่มวิเคราะห์ความเสี่ยง", use_container_width=True)
with btn_2:
    if st.button("🔄 รีเซ็ตข้อมูล", use_container_width=True): 
        st.rerun()

# ส่วนการแสดงผลลัพธ์การคาดการณ์
if submit:
    input_data = np.array([[sbp, dbp, heart_rate, bmi, pedigree, age, sleep_hours, pct_sodium, exercise, sleep_apnea]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("### 📝 รายงานการวิเคราะห์ผลสุขภาพเชิงรับ:")
    
    if prediction == 1 or sbp >= 140 or dbp >= 90:
        st.error("🔴 **ผลการประเมินโดย AI: ตรวจพบสภาวะความเสี่ยงสูงต่อภาวะโรคความดันโลหิตสูง**")
        
        with st.expander("🔍 คลิกเพื่อดูคำแนะนำการปรับเปลี่ยนพฤติกรรมเฉพาะบุคคล", expanded=True):
            if sleep_apnea == 1:
                st.warning("⚠️ **ปัจจัยด้านการนอนหลับ (Obstructive Sleep Apnea):** อาการนอนกรนสะดุดส่งผลให้ระดับออกซิเจนในเลือดวิกฤตตกลงในตอนกลางคืน แนะนำให้พบแพทย์เฉพาะทางเพื่อทำ Sleep Test")
            if exercise == 0:
                st.warning("⚠️ **ปัจจัยด้านกิจกรรมทางกาย (Sedentary Behavior):** การขาดกิจกรรมทางกายลดความยืดหยุ่นของระบบหลอดเลือด แนะนำให้เดินเร็วต่อเนื่องสะสมให้ได้ 150 นาที/สัปดาห์")
            if pct_sodium > 50:
                st.info("💡 **คำแนะนำการบริโภค:** การทานโซเดียมสูงส่งผลต่อการกักเก็บน้ำในระบบหลอดเลือด ควรลดอาหารรสเค็มจัดหรืออาหารแปรรูป")
    else:
        st.success("🟢 **ผลการประเมินโดย AI: สัญญาณชีพและโครงสร้างพฤติกรรมอยู่ในเกณฑ์ปกติ (เสี่ยงต่ำ)**")
        st.balloons()
        st.info("🎉 **ข้อแนะนำระดับสากล:** คุณมีระดับสัญญาณชีพที่สมดุลดีเยี่ยม รักษาวินัยการนอนและการขยับร่างกายแบบนี้ต่อไปยาวๆ ครับ")

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.caption("ℹ️ **Medical Disclaimer:** ระบบปัญญาประดิษฐ์นี้ทำหน้าที่คัดกรองแนวโน้มความเสี่ยงในเบื้องต้นเท่านั้น ไม่สามารถใช้ทดแทนการวินิจฉัยโรคโดยแพทย์และเครื่องมือมาตรฐานในสถานพยาบาลได้")