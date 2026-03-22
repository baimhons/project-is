import streamlit as st

st.page_link("app.py", label="หน้าหลัก")
st.title("Model Ensemble")

# ==========================================
# ที่มาของ Dataset
# ==========================================
st.header("การจัดเตรียม Dataset")
st.markdown("""
ใช้ **ChatGPT** ช่วย Generate Dataset โดยระบุโครงสร้างและประเภทของข้อมูลก่อน ได้แก่

| Column | ประเภท | คำอธิบาย |
|---|---|---|
| 'Building_ID' | Text | รหัสอาคาร |
| 'Age_Years' | Numeric | อายุอาคาร (ปี) |
| 'Crack_Length_mm' | Numeric | ความยาวรอยแตก (มม.) |
| 'Crack_Width_mm' | Numeric | ความกว้างรอยแตก (มม.) |
| 'Material' | Categorical | วัสดุก่อสร้าง |
| 'Severity_Level' | Label | ระดับความรุนแรง (Low / Medium / High / Critical) |

> Dataset ที่ Generate มาจะมีความสมจริงแต่ยังมี **ข้อมูลที่ไม่สมบูรณ์** และ **ค่าผิดปกติ** ปะปนอยู่
""")

st.divider()

# ==========================================
# ขั้นตอนการพัฒนา
# ==========================================
st.header("ขั้นตอนการพัฒนา")

st.subheader("1. Data Cleansing")
st.markdown("""
ก่อน train model ต้องทำความสะอาดข้อมูลก่อน เพราะ Dataset ที่ได้มามีปัญหาหลายอย่าง
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**ปัญหาที่พบ**")
    st.markdown("""
- 'Material' มีค่าสกปรก เช่น '"Con-crete"', 'Con crete'
- 'Crack_Length_mm' เป็น string มีขยะ เช่น 'Unknown'
- 'Age_Years' ติดลบ เช่น '-5', '-10'
- ค่า outlier สุดขีด เช่น 'Age = 1000', 'Crack = 9999'
- 'Severity_Level = Extreme' มีแค่ 1 แถว
- Missing values หลาย column
""")

with col2:
    st.markdown("**วิธีแก้ไข**")
    st.markdown("""
- Normalize ชื่อวัสดุให้เป็นมาตรฐาน
- แปลง string → float และลบค่าที่แปลงไม่ได้
- ค่าติดลบหรือเกิน threshold → แทนด้วย NaN
- ตัด row ที่เป็น outlier หรือ label หายากออก
- เติม Numeric ด้วย **Median**, Categorical ด้วย **Mode**
""")

st.subheader("2. Feature Engineering")
st.markdown("""
แปลงข้อมูล Text ให้เป็นตัวเลขก่อน train เพราะ ML model รับได้เฉพาะตัวเลข

- **'Material'** → แปลงด้วย 'LabelEncoder' เช่น 'Concrete=1', 'Brick=0'
- **'Severity_Level'** → แปลงด้วย 'LabelEncoder' เช่น 'Low=2', 'Critical=0'
- แบ่ง **Train 80% / Test 20%** โดยใช้ 'stratify=y' เพื่อให้สัดส่วน class สมดุล
""")

st.subheader("3. Train 3 Models")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Random Forest**")
    st.markdown("""
- ไม่ต้อง Scale feature
- สร้าง Decision Tree หลายต้นแล้วโหวตผล
- Tune: 'n_estimators=100'
- บอก Feature Importance ได้
""")
with col2:
    st.markdown("**KNN**")
    st.markdown("""
- ต้อง **Scale** ก่อนเสมอ
- ดูจาก K จุดที่ใกล้ที่สุดแล้วโหวต
- หา Best K ด้วย Cross-Validation
- Save 'knn_scaler.pkl' แยก
""")
with col3:
    st.markdown("**SVM**")
    st.markdown("""
- ต้อง **Scale** ก่อนเสมอ
- หา Hyperplane ที่แบ่ง class ได้ดีที่สุด
- Tune 'C', 'kernel', 'gamma' ด้วย GridSearchCV
- Save 'svm_scaler.pkl' แยก
""")

st.subheader("4. Evaluate & Compare")
st.markdown("""
วัดผลแต่ละ model ด้วย test set ชุดเดิม (random_state=42) เพื่อให้เปรียบเทียบได้ยุติธรรม

- **Accuracy** — ภาพรวมว่าทำนายถูกกี่ %
- **Precision / Recall / F1** — วัดละเอียดแยกตาม class
- **Confusion Matrix** — ดูว่า model สับสน class ไหนกับ class ไหน
""")

st.divider()

# ==========================================
# ประโยชน์จากการ Train
# ==========================================
st.header("ประโยชน์จากการ Train Model")

st.markdown("""
โมเดลที่ train มาทั้ง 3 ตัวมีวัตถุประสงค์เพื่อ **พยากรณ์ระดับความรุนแรงของความเสียหายในอาคาร**
โดยรับข้อมูลจากการตรวจสอบอาคารและส่งออกเป็น Severity Level
""")

col1, col2 = st.columns(2)
with col1:
    st.info("**- ช่วยวิศวกรตัดสินใจ**\n\nแทนที่ผู้เชี่ยวชาญต้องประเมินทุกอาคารด้วยตัวเอง model ช่วย triage เบื้องต้นได้ทันที")
    st.info("**- จัดลำดับการซ่อมบำรุง**\n\nกรองอาคาร Critical ออกมาก่อน เพื่อจัดสรรทรัพยากรและงบประมาณได้ถูกจุด")
with col2:
    st.info("**- ระบบ Early Warning**\n\nเมื่อมีข้อมูลตรวจสอบใหม่เข้ามา model แจ้งเตือนระดับความเสี่ยงได้ทันทีโดยไม่ต้องรอผู้เชี่ยวชาญ")
    st.info("**- เปรียบเทียบประสิทธิภาพ**\n\nการ train หลาย model ช่วยเลือก algorithm ที่เหมาะสมที่สุดกับข้อมูลประเภทนี้")

st.divider()

# ==========================================
# Feature ที่สำคัญที่สุด
# ==========================================
st.header("ปัจจัยที่ส่งผลต่อ Severity มากที่สุด")
st.markdown("""จาก **Feature Importance ของ Random Forest** พบว่า:""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("1. Crack Width",    "~40%", "สำคัญที่สุด")
col2.metric("2. Crack Length",   "~39%", "สำคัญมาก")
col3.metric("3. Age (ปี)",       "~20%", "ปานกลาง")
col4.metric("4️. Material",       "~2%",  "น้อยมาก")

st.caption("ความกว้างและความยาวรอยแตกเป็นตัวบ่งชี้หลักของความรุนแรง ส่วนวัสดุก่อสร้างมีผลน้อยมากในชุดข้อมูลนี้")