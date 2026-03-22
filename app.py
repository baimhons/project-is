import streamlit as st

st.title("Project Intelligent System")

st.space("medium")

st.text("Ensemble models มี Random Forest, K Nearest Neighbor, Support Vector Machines")

st.title("อธิบายการทำงานแต่ละขั้นตอน")

st.page_link("pages/Model_Ensemble_Explain.py", label="อธิบายการพัฒนา Model Ensemble" )
st.page_link("pages/Neural_Network_Explain.py", label="อธิบายการพัฒนา Neural Network" )

st.space("medium")

st.title("การทดสอบ Models")

st.page_link("pages/Model_Ensemble.py", label="การ Train Model Ensemble" )
st.page_link("pages/Neural_Network.py", label="การ Train Neural Network" )