import streamlit as st
import numpy as np
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Hypertension Screen", page_icon="🟢", layout="centered")

# ตกแต่งสไตล์ CSS โทนสีเขียวสุขภาพ
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
    div.stButton > button:first-child:hover {
        background-color: #1b5e20;
    }
    hr { border-top: 2px solid #c8e6c9; }
    h1 { color: #1b5e20; }
    h2, h3 { color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

# โหลดสมองกล AI จากไฟล์ความดันโลหิตสูง
@st.cache_resource
def load_model():
    data = joblib.load("hypertension_svm_system.pkl")
    return data["model"], data["scaler"]

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ 'hypertension_svm_system.pkl' กรุณาตรวจสอบการอัปโหลดไฟล์ลงใน GitHub")
    st.stop()

# ส่วนหัวข้อหลัก
st.title("🟢 ระบบคัดกรองความเสี่ยงโรคความดันโลหิตสูงขั้นสูง")
st.write("ประเมินสภาวะหลอดเลือดร่วมกับพฤติกรรมการบริโภคด้วยปัญญาประดิษฐ์ (AI SVM)")
st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนอินพุตข้อมูล
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

st.subheader("🧠 2. ประวัติพันธุกรรมและความเครียด")
col_c, col_d = st.columns(2)
with col_c:
    family_history = st.selectbox("มีญาติสายตรงเป็นโรคความดันโลหิตสูงหรือไม่?", ["ไม่มีประวัติในครอบครัว", "มีพ่อหรือแม่เป็น", "ทั้งพ่อและแม่เป็น"])
    pedigree = 0.20 if family_history == "ไม่มีประวัติในครอบครัว" else (0.50 if family_history == "มีพ่อหรือแม่เป็น" else 0.80)
    stress_level = st.selectbox("ระดับความเครียดในปัจจุบัน", ["ต่ำ / ผ่อนคลายปกติ", "ปานกลาง", "สูง / เครียดสะสม"])
with col_d:
    sleep_hours = st.slider("จำนวนชั่วโมงการนอนหลับเฉลี่ยต่อคืน", min_value=3, max_value=12, value=7)

st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("🧂 3. พฤติกรรมการกินและสัญญาณชีพ")
col1, col2 = st.columns(2)
with col1:
    pct_sodium = st.slider("สัดส่วนการกินอาหารเค็ม / แปรรูป / โซเดียมสูง (%)", min_value=0, max_value=100, value=35)
    sbp = st.number_input("ความดันโลหิตตัวบน (Systolic BP) [mm Hg]", min_value=70, max_value=230, value=120)
with col2:
    heart_rate = st.number_input("อัตราการเต้นของหัวใจขณะพัก [ครั้ง/นาที]", min_value=40, max_value=180, value=75)
    dbp = st.number_input("ความดันโลหิตตัวล่าง (Diastolic BP) [mm Hg]", min_value=40, max_value=140, value=80)

st.markdown("<hr>", unsafe_allow_html=True)

# ส่วนการประมวลผลลัพธ์
btn_1, btn_2 = st.columns([4, 1])
with btn_1:
    submit = st.button("🔮 เริ่มวิเคราะห์ความเสี่ยงองค์รวมเชิงลึก", use_container_width=True)
with btn_2:
    if st.button("🔄 ล้างข้อมูล", use_container_width=True):
        st.rerun()

if submit:
    input_data = np.array([[sbp, dbp, heart_rate, bmi, pedigree, age, sleep_hours, pct_sodium]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
     
    st.markdown("### 📝 รายงานผลการประเมิน:")
    if prediction == 1 or sbp >= 140 or dbp >= 90:
        st.error("🔴 **ผลลัพธ์จาก AI: มีความเสี่ยงสูงต่อภาวะโรคความดันโลหิตสูง**")
        st.markdown("<div style='background-color:#ffebee; padding:15px; border-left:6px solid #c62828; border-radius:4px; color:#c62828;'>⚠️ แนะนำให้ตรวจวัดซ้ำหลังจากนั่งพักผ่อน และควรปรึกษาแพทย์ผู้เชี่ยวชาญหากค่ายังคงสูงเกินมาตรฐาน</div>", unsafe_allow_html=True)
    else:
        st.success("🟢 **ผลลัพธ์จาก AI: สัญญาณชีพและพฤติกรรมอยู่ในเกณฑ์เสี่ยงต่ำ (ปกติ)**")
        st.markdown("<div style='background-color:#e8f5e9; padding:15px; border-left:6px solid #2e7d32; border-radius:4px; color:#2e7d32;'>🎉 สุขภาพระบบหลอดเลือดและพฤติกรรมโดยรวมอยู่ในเกณฑ์ดี รักษาวินัยการดูแลตัวเองต่อไปนะครับ</div>", unsafe_allow_html=True)

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.caption("ℹ️ **Medical Disclaimer:** ระบบนี้เป็นเพียงการคัดกรองความเสี่ยงเบื้องต้นด้วยแบบจำลองคณิตศาสตร์ AI เท่านั้น ไม่สามารถใช้ทดแทนการวินิจฉัยโรคอย่างเป็นทางการโดยแพทย์ในสถานพยาบาลได้")