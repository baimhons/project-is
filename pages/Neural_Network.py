import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.page_link("app.py", label="หน้าหลัก")
st.title("🧠 Neural Network")
st.markdown("**Customer Purchase Prediction** — ทำนายว่าลูกค้าจะกลับมาซื้อซ้ำหรือไม่")

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("models/customer_purchase_nn.keras")
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_model()

if load_error:
    st.error(f"❌ โหลด model ไม่ได้: {load_error}")
    st.info("💡 ตรวจสอบว่าไฟล์ `customer_purchase_nn.keras` อยู่ใน folder เดียวกับ app นี้")
    st.stop()

st.success("✅ โหลด model สำเร็จ")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 ทำนายลูกค้า", "📋 Model Info"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("กรอกข้อมูลลูกค้า")

    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input(
            "💰 รายได้ต่อเดือน (บาท)",
            min_value=0, max_value=500_000,
            value=25000, step=1000
        )
        time_on_app = st.number_input(
            "📱 เวลาใช้งานแอป (นาที/ครั้ง)",
            min_value=0.0, max_value=600.0,
            value=20.0, step=0.5
        )

    with col2:
        last_purchase_days = st.number_input(
            "📅 วันที่ซื้อล่าสุด (กี่วันที่แล้ว)",
            min_value=0, max_value=365,
            value=14, step=1
        )
        device = st.selectbox(
            "📲 อุปกรณ์ที่ใช้",
            options=["Android", "iOS", "Windows"]
        )

    # Encode device
    device_android = 1 if device == "Android" else 0
    device_ios     = 1 if device == "iOS" else 0
    device_windows = 1 if device == "Windows" else 0

    # Feature order ต้องตรงกับ training
    features = np.array([[
        monthly_income,
        time_on_app,
        last_purchase_days,
        device_android,
        device_ios,
        device_windows,
    ]], dtype=np.float32)

    # Scaling (ใช้ค่า median จาก training data เป็น fallback)
    FEATURE_MEANS = np.array([28000.0, 22.5, 18.0, 0.5, 0.35, 0.15])
    FEATURE_STDS  = np.array([25000.0, 30.0, 40.0, 0.5, 0.48, 0.36])
    features_scaled = (features - FEATURE_MEANS) / FEATURE_STDS

    st.divider()

    if st.button("🚀 ทำนาย", type="primary", use_container_width=True):
        prob = float(model.predict(features_scaled, verbose=0)[0][0])
        pred = "Will Buy 🟢" if prob >= 0.5 else "Will NOT Buy 🔴"

        # Result card
        st.markdown("### ผลการทำนาย")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if prob >= 0.5:
                st.success(f"## {pred}")
            else:
                st.error(f"## {pred}")

        with col_r2:
            st.metric("ความน่าจะเป็น (Will Buy)", f"{prob*100:.1f}%")

        # Probability gauge
        st.markdown("**Probability Bar**")
        bar_col1, bar_col2 = st.columns([prob, max(1-prob, 0.001)])
        with bar_col1:
            st.markdown(
                f'<div style="background:#2ecc71;border-radius:8px;padding:10px;'
                f'text-align:center;color:white;font-weight:bold;">'
                f'Buy {prob*100:.1f}%</div>',
                unsafe_allow_html=True
            )
        with bar_col2:
            st.markdown(
                f'<div style="background:#e74c3c;border-radius:8px;padding:10px;'
                f'text-align:center;color:white;font-weight:bold;">'
                f'Not {(1-prob)*100:.1f}%</div>',
                unsafe_allow_html=True
            )

        # Summary table
        st.markdown("**ข้อมูลที่ใช้ทำนาย**")
        summary = pd.DataFrame({
            "Feature": ["Monthly Income", "Time on App (min)", "Last Purchase (days)", "Device"],
            "Value": [f"{monthly_income:,} บาท", f"{time_on_app} นาที", f"{last_purchase_days} วัน", device]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📋 Model Architecture")

    # Model summary as text
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    st.code("\n".join(summary_lines), language="text")

    st.divider()
    st.subheader("🏗️ Architecture Diagram")

    # Draw architecture diagram with matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    layers_info = [
        ("Input\n(6)", 1.0),
        ("Dense 64\n+ BN + ReLU\n+ Dropout", 3.0),
        ("Dense 32\n+ BN + ReLU\n+ Dropout", 5.0),
        ("Dense 16\nReLU", 7.0),
        ("Output\nSigmoid", 9.0),
    ]
    colors = ['#3498db', '#9b59b6', '#9b59b6', '#e67e22', '#2ecc71']

    for (label, x), color in zip(layers_info, colors):
        rect = plt.Rectangle((x - 0.6, 1.5), 1.2, 2, color=color, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(x, 2.5, label, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=4)

    for i in range(len(layers_info) - 1):
        x1 = layers_info[i][1] + 0.6
        x2 = layers_info[i+1][1] - 0.6
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color='white', lw=1.5))

    ax.set_title("Neural Network Architecture", color='white', fontsize=13, pad=10)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("ℹ️ Model Details")
    col1, col2, col3 = st.columns(3)
    col1.metric("Input Features", "6")
    col2.metric("Hidden Layers", "3")
    col3.metric("Output", "Binary (0/1)")

    st.markdown("""
    | Layer | Units | Activation | Regularization |
    |-------|-------|------------|----------------|
    | Dense 1 | 64 | ReLU | BatchNorm + Dropout |
    | Dense 2 | 32 | ReLU | BatchNorm + Dropout |
    | Dense 3 | 16 | ReLU | — |
    | Output | 1 | Sigmoid | — |

    **Loss:** Binary Crossentropy &nbsp;|&nbsp; **Optimizer:** Adam &nbsp;|&nbsp; **Metrics:** Accuracy, AUC, Precision, Recall
    """)