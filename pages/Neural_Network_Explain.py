import streamlit as st

st.page_link("app.py", label="หน้าหลัก")
st.title("Neural Network")

# ==========================================
# ที่มาของ Dataset
# ==========================================
st.header("การจัดเตรียม Dataset")
st.markdown("""
ใช้ **ChatGPT** ช่วย Generate Dataset โดยระบุโครงสร้างและประเภทของข้อมูลก่อน ได้แก่

| Column | ประเภท | คำอธิบาย |
|---|---|---|
| `Cust_ID` | Text | รหัสลูกค้า |
| `Monthly_Income` | Numeric | รายได้ต่อเดือน (บาท) |
| `Time_on_App_min` | Numeric | เวลาใช้งานแอปต่อครั้ง (นาที) |
| `Last_Purchase_Days` | Numeric | จำนวนวันนับจากการซื้อครั้งล่าสุด |
| `Device` | Categorical | อุปกรณ์ที่ใช้ (Android / iOS / Windows) |
| `Will_Buy_Again` | Label | ลูกค้าจะกลับมาซื้อซ้ำหรือไม่ (Yes / No) |

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
- `Monthly_Income` มีขยะ เช่น `'30000 Baht'`, `'40000THB'`
- ค่าติดลบที่ไม่สมเหตุสมผล เช่น `Monthly_Income = -5000`
- ค่า outlier สุดขีด เช่น `Income = 9999999`, `Time = 9999`
- `Time_on_App_min` มีค่า string เช่น `'Unknown'`, `'abc'`
- `Last_Purchase_Days` มีค่าที่ไม่ถูกต้อง เช่น `'?'`, `-10`
- `Device` มี typo เช่น `'"Window"'`, `'Window'` (ซ้ำกัน)
- Missing values หลาย column
""")

with col2:
    st.markdown("**วิธีแก้ไข**")
    st.markdown("""
- ดึงเฉพาะตัวเลขออกจาก string ด้วย regex
- ค่าติดลบหรือเกิน threshold → แทนด้วย `NaN`
- แปลง string → float และลบค่าที่แปลงไม่ได้
- Normalize ชื่ออุปกรณ์ให้เป็นมาตรฐาน
- เติม Numeric ด้วย **Median**, Categorical ด้วย **Mode**
""")

st.subheader("2. Feature Engineering")
st.markdown("""
แปลงข้อมูล Categorical ให้เป็นตัวเลขก่อน train เพราะ Neural Network รับได้เฉพาะตัวเลข

- **`Device`** → แปลงด้วย `One-Hot Encoding` ได้ 3 columns (`Device_Android`, `Device_iOS`, `Device_Windows`)
- **`Will_Buy_Again`** → แปลง `Yes=1`, `No=0`
- แบ่ง **Train 80% / Test 20%** โดยใช้ `stratify=y` เพื่อให้สัดส่วน class สมดุล
- **StandardScaler** — Normalize ค่าตัวเลขทุก feature ให้อยู่ใน scale เดียวกัน เพื่อให้ Neural Network เรียนรู้ได้เร็วและแม่นยำขึ้น
""")

st.subheader("3. Neural Network Architecture")
st.markdown("ออกแบบโครงสร้าง Neural Network แบบ **Feedforward** สำหรับ Binary Classification")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Input Layer**")
    st.markdown("""
- รับ 6 features
- ผ่าน StandardScaler แล้ว
""")
with col2:
    st.markdown("**Hidden Layer 1**")
    st.markdown("""
- 64 neurons
- Activation: **ReLU**
- BatchNormalization
- Dropout (0.3)
""")
with col3:
    st.markdown("**Hidden Layer 2**")
    st.markdown("""
- 32 neurons
- Activation: **ReLU**
- BatchNormalization
- Dropout (0.3)
""")
with col4:
    st.markdown("**Output Layer**")
    st.markdown("""
- 1 neuron
- Activation: **Sigmoid**
- ทำนายค่า 0–1
- threshold = 0.5
""")

st.subheader("4. Training Configuration")
st.markdown("""
กำหนดค่าต่าง ๆ สำหรับการ train model
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Hyperparameters**")
    st.markdown("""
- **Loss Function:** `Binary Crossentropy`
- **Optimizer:** `Adam` (lr = 0.001)
- **Metrics:** Accuracy, AUC, Precision, Recall
- **Batch Size:** 16
- **Max Epochs:** 150
""")
with col2:
    st.markdown("**Callbacks**")
    st.markdown("""
- **EarlyStopping** — หยุด train เมื่อ `val_auc` ไม่ดีขึ้นใน 15 epoch ติดต่อกัน และโหลด weights ที่ดีที่สุดกลับมา
- **ReduceLROnPlateau** — ลด Learning Rate ลงครึ่งหนึ่งเมื่อ `val_loss` ไม่ลดลงใน 8 epoch
""")

st.subheader("5. Evaluate & Visualize")
st.markdown("""
วัดผล model ด้วย test set และแสดงผลในหลายมิติ

- **Accuracy / AUC / Precision / Recall** — วัดประสิทธิภาพภาพรวม
- **Confusion Matrix** — ดูว่า model สับสน class ไหน
- **ROC Curve** — วัดความสามารถในการแยก class ทั้งสอง
- **Probability Distribution** — ดูการกระจายของค่า prediction
""")

st.divider()

# ==========================================
# ประโยชน์จากการ Train
# ==========================================
st.header("ประโยชน์จากการ Train Model")

st.markdown("""
โมเดล Neural Network ที่ train มามีวัตถุประสงค์เพื่อ **พยากรณ์พฤติกรรมการซื้อซ้ำของลูกค้า**
โดยรับข้อมูลพฤติกรรมของลูกค้าและส่งออกเป็นความน่าจะเป็นที่จะกลับมาซื้อซ้ำ
""")

col1, col2 = st.columns(2)
with col1:
    st.info("**📊 วิเคราะห์กลุ่มลูกค้า**\n\nระบุลูกค้าที่มีโอกาสหายไป (Churn) ได้ล่วงหน้า เพื่อให้ทีม Marketing วางแผนรักษาลูกค้าได้ทันท่วงที")
    st.info("**🎯 เพิ่มประสิทธิภาพ Campaign**\n\nส่ง Promotion เฉพาะกลุ่มลูกค้าที่มีโอกาสซื้อซ้ำสูง ลดต้นทุนและเพิ่ม Conversion Rate")
with col2:
    st.info("**🔔 ระบบแจ้งเตือนอัตโนมัติ**\n\nเมื่อลูกค้าไม่ได้ใช้งานนานเกินกำหนด model สามารถ flag และส่ง notification โดยอัตโนมัติ")
    st.info("**📈 วางแผนสต็อกและบริการ**\n\nคาดการณ์จำนวนลูกค้าที่จะกลับมาในช่วงเวลาหนึ่ง ช่วยจัดการทรัพยากรได้แม่นยำขึ้น")

st.divider()

# ==========================================
# Feature ที่สำคัญที่สุด
# ==========================================
st.header("ปัจจัยที่ส่งผลต่อการซื้อซ้ำมากที่สุด")
st.markdown("""จากการวิเคราะห์ข้อมูลและผลการ train model พบว่า:""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("1. Last Purchase Days", "สำคัญที่สุด", "ยิ่งนานยิ่งเสี่ยง Churn")
col2.metric("2. Time on App",        "สำคัญมาก",   "ใช้งานนานกว่า = ผูกพันกว่า")
col3.metric("3. Monthly Income",     "ปานกลาง",    "รายได้สูงซื้อบ่อยกว่า")
col4.metric("4. Device",             "น้อยมาก",    "ไม่ค่อยส่งผลต่อการตัดสินใจ")

st.caption("ระยะเวลาห่างจากการซื้อครั้งล่าสุดและเวลาที่ใช้บนแอปเป็นสัญญาณสำคัญที่สุดของพฤติกรรมการซื้อซ้ำ")