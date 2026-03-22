import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

st.page_link("app.py", label="หน้าหลัก")

# ==========================================
# Load Models & Data
# ==========================================
@st.cache_resource
def load_models():
    rf         = joblib.load("models/random_forest_model.pkl")
    knn        = joblib.load("models/knn_model.pkl")
    knn_scaler = joblib.load("models/knn_scaler.pkl")
    svm        = joblib.load("models/svm_model.pkl")
    svm_scaler = joblib.load("models/svm_scaler.pkl")
    le_m       = joblib.load("models/le_material.pkl")
    le_t       = joblib.load("models/le_target.pkl")
    return rf, knn, knn_scaler, svm, svm_scaler, le_m, le_t

@st.cache_data
def load_data():
    return pd.read_csv("data/dataset_building_cleaned.csv")

try:
    rf_model, knn_model, knn_scaler, svm_model, svm_scaler, le_material, le_target = load_models()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"ไม่พบไฟล์: {e}")
    st.stop()

feature_cols = ["Age_Years", "Crack_Length_mm", "Crack_Width_mm", "Material_enc"]

# เตรียม X, y
df["Material_enc"] = le_material.transform(df["Material"])
df["Target"]       = le_target.transform(df["Severity_Level"])
X = df[feature_cols]
y = df["Target"]

_, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict ทั้ง 3 model
y_pred_rf  = rf_model.predict(X_test_raw)
y_pred_knn = knn_model.predict(knn_scaler.transform(X_test_raw))
y_pred_svm = svm_model.predict(svm_scaler.transform(X_test_raw))

acc_rf  = accuracy_score(y_test, y_pred_rf)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svm = accuracy_score(y_test, y_pred_svm)

# ==========================================
# Helper: Confusion Matrix
# ==========================================
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

# ==========================================
# SECTION 1: Random Forest
# ==========================================
st.title("Random Forest")

tab1, tab2 = st.tabs(["- Feature Importance", "- Model Report"])

with tab1:
    st.subheader("Feature Importances")
    importances = (
        pd.Series(rf_model.feature_importances_, index=feature_cols)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.barh(importances.index, importances.values, color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importances")
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    st.dataframe(
        importances[::-1].reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"}),
        hide_index=True, use_container_width=True,
    )

with tab2:
    st.subheader("Model Evaluation")
    st.metric("Accuracy", f"{acc_rf * 100:.2f}%")
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred_rf, target_names=le_target.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)
    st.markdown("#### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(y_test, y_pred_rf, le_target.classes_), use_container_width=True)

# ==========================================
# SECTION 2: KNN
# ==========================================
st.title("K Nearest Neighbor")

tab3, tab4 = st.tabs(["- K Selection", "- Model Report"])

with tab3:
    st.subheader("K ที่ใช้ Train")
    best_k = knn_model.n_neighbors
    st.metric("Best K", best_k)
    st.info(f"KNN เลือก K ที่ดีที่สุดด้วย Cross-Validation (5-fold) โดย K={best_k} ให้ accuracy สูงสุดบน training set")
    st.subheader("Feature Scaling")
    st.markdown(
        "KNN ใช้ **ระยะห่าง** ในการตัดสินใจ → ต้อง scale feature ก่อนเสมอ  \n"
        "ใช้ `StandardScaler` แปลงทุก feature ให้มี **mean=0, std=1**"
    )
    sample = X_test_raw.iloc[:5].copy()
    sample_scaled = knn_scaler.transform(sample)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Scaling**")
        st.dataframe(sample.round(2), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**After Scaling**")
        st.dataframe(pd.DataFrame(sample_scaled, columns=feature_cols).round(4),
                     hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Model Evaluation")
    st.metric("Accuracy", f"{acc_knn * 100:.2f}%")
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred_knn, target_names=le_target.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)
    st.markdown("#### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(y_test, y_pred_knn, le_target.classes_), use_container_width=True)

# ==========================================
# SECTION 3: SVM
# ==========================================
st.title("Support Vector Machine")

tab5, tab6 = st.tabs(["- Hyperparameters", "- Model Report"])

with tab5:
    st.subheader("Best Hyperparameters จาก GridSearchCV")
    params = svm_model.get_params()
    st.metric("Kernel", params["kernel"])
    col1, col2 = st.columns(2)
    col1.metric("C (Regularization)", params["C"])
    col2.metric("Gamma", params["gamma"])
    st.info(
        "SVM ใช้ **GridSearchCV** ลอง C × kernel × gamma ทุก combination แล้วเลือก "
        "ตัวที่ให้ accuracy สูงสุดบน 5-fold cross-validation"
    )
    st.subheader("Feature Scaling")
    st.markdown(
        "SVM คำนวณ **hyperplane** โดยใช้ระยะห่าง → ต้อง scale feature เช่นเดียวกับ KNN  \n"
        "ใช้ `StandardScaler` แยกต่างหากจาก KNN"
    )

with tab6:
    st.subheader("Model Evaluation")
    st.metric("Accuracy", f"{acc_svm * 100:.2f}%")
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred_svm, target_names=le_target.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)
    st.markdown("#### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(y_test, y_pred_svm, le_target.classes_), use_container_width=True)

# ==========================================
# SECTION 4: เปรียบเทียบ 3 Model
# ==========================================
st.title("เปรียบเทียบ 3 Model")

best_acc = max(acc_rf, acc_knn, acc_svm)
col1, col2, col3 = st.columns(3)
col1.metric("Random Forest",          f"{acc_rf  * 100:.2f}%", delta="🏆 Best" if acc_rf  == best_acc else None)
col2.metric(f"KNN (K={best_k})",      f"{acc_knn * 100:.2f}%", delta="🏆 Best" if acc_knn == best_acc else None)
col3.metric(f"SVM ({params['kernel']})", f"{acc_svm * 100:.2f}%", delta="🏆 Best" if acc_svm == best_acc else None)

model_names = ["Random Forest", f"KNN (K={best_k})", f"SVM ({params['kernel']})"]
accs = [acc_rf, acc_knn, acc_svm]
bar_colors = ["#55A868", "#4C72B0", "#C44E52"]

fig_cmp, ax_cmp = plt.subplots(figsize=(6, 4))
bars = ax_cmp.bar(model_names, accs, color=bar_colors)
ax_cmp.set_ylim(0, 1.15)
ax_cmp.set_ylabel("Accuracy")
ax_cmp.set_title("Model Comparison: RF vs KNN vs SVM")
for bar, val in zip(bars, accs):
    ax_cmp.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val * 100:.2f}%", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
st.pyplot(fig_cmp, use_container_width=True)

# ตารางสรุป
summary_df = pd.DataFrame({
    "Model":       model_names,
    "Accuracy (%)": [round(a * 100, 2) for a in accs],
    "ต้อง Scale":  ["❌", "✅", "✅"],
    "Tune Param":  ["n_estimators", "K (n_neighbors)", "C, kernel, gamma"],
})
st.dataframe(summary_df, hide_index=True, use_container_width=True)