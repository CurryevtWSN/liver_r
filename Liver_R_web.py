#%% load package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from tabpfn import TabPFNClassifier

# ✅ 必须尽量放在最前面
st.set_page_config(
    page_title='Interpretable Prediction of Post-Transplant Recurrence in Hepatocellular Carcinoma Using TabPFN and SHAP'
)

# 如需切换工作目录（建议用绝对路径或用 __file__ 更稳）
os.chdir(r"D:\data_analysis\machine_learning\肝癌肝移植\肝癌肝移植")

#%% title
st.title('Interpretable Prediction of Post-Transplant Recurrence in Hepatocellular Carcinoma Using TabPFN and SHAP')

#%% sidebar inputs
st.sidebar.markdown('## Variables')

Age = st.sidebar.slider("Age (year)", 0, 100, value=65, step=1)

Portal_Hepatic_Vein_Tumor_Thrombus = st.sidebar.selectbox(
    "Portal_Hepatic_Vein_Tumor_Thrombus", ("No", "Yes"), index=0
)

Preoperative_BCLC_Stage = st.sidebar.selectbox(
    "Preoperative_BCLC_Stage", (0, 1, 2, 3, 4), index=1
)

Preoperative_Tumor_Number = st.sidebar.slider(
    "Preoperative_Tumor_Number", 1, 3, value=1, step=1
)

Maximum_Tumor_Diameter = st.sidebar.slider(
    "Maximum_Tumor_Diameter", 0.00, 15.00, value=5.00, step=0.01, format="%.2f"
)

Preoperative_AFP = st.sidebar.slider(
    "Preoperative_AFP", 0.0, 10000.0, value=10.0, step=0.1, format="%.1f"
)

Preoperative_GGT = st.sidebar.slider(
    "Preoperative_GGT", 0.0, 2000.0, value=50.0, step=0.1, format="%.1f"
)

Preoperative_ALB = st.sidebar.slider(
    "Preoperative_ALB", 0.0, 60.0, value=38.2, step=0.1, format="%.1f"
)

Preoperative_TB = st.sidebar.slider(
    "Preoperative_TB", 0.0, 500.0, value=15.0, step=0.1, format="%.1f"
)

Preoperative_NLR = st.sidebar.slider(
    "Preoperative_NLR", 0.00, 100.00, value=2.00, step=0.01, format="%.2f"
)

Preoperative_INR = st.sidebar.slider(
    "Preoperative_INR", 0.0000, 5.0000, value=1.0000, step=0.0001, format="%.4f"
)

Preoperative_Neutrophil = st.sidebar.slider(
    "Preoperative_Neutrophil", 0.00, 100.00, value=3.00, step=0.01, format="%.2f"
)

Preoperative_Lymphocyte = st.sidebar.slider(
    "Preoperative_Lymphocyte", 0.00, 100.00, value=1.50, step=0.01, format="%.2f"
)

Preoperative_HBsAg = st.sidebar.slider(
    "Preoperative_HBsAg", 0.0, 6000.0, value=0.0, step=0.1, format="%.1f"
)

st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved')

#%% encode categorical
yesno_map = {'No': 0, 'Yes': 1}
Portal_Hepatic_Vein_Tumor_Thrombus = yesno_map[Portal_Hepatic_Vein_Tumor_Thrombus]

#%% load model & data
data_train = pd.read_csv('train.csv')

# data = data_train
features = ["Maximum_Tumor_Diameter","Preoperative_AFP","Preoperative_BCLC_Stage","Preoperative_GGT",
            "Preoperative_Tumor_Number","Preoperative_ALB","Portal_Hepatic_Vein_Tumor_Thrombus",
            "Preoperative_TB","Preoperative_NLR","Preoperative_INR","Preoperative_Neutrophil",
            "Preoperative_Lymphocyte","Age","Preoperative_HBsAg"]

indicator = ["Recurrence"]

X_train = data_train[features]
y_train= data_train[indicator]

X_train = X_train.values
y_train = y_train.values

TabPFN = TabPFNClassifier(model_path = "tabpfn-v2-classifier.ckpt")
TabPFN_model = TabPFN.fit(X_train, y_train)

hp_train = pd.read_csv('train.csv')  # 你后面暂时没用到，可以保留

#%% prediction input
X_input = np.array([[
    Age,
    Portal_Hepatic_Vein_Tumor_Thrombus,
    Preoperative_BCLC_Stage,
    Preoperative_Tumor_Number,
    Maximum_Tumor_Diameter,
    Preoperative_AFP,
    Preoperative_GGT,
    Preoperative_ALB,
    Preoperative_TB,
    Preoperative_NLR,
    Preoperative_INR,
    Preoperative_Neutrophil,
    Preoperative_Lymphocyte,
    Preoperative_HBsAg
]])

# 阈值
sp = 0.5

# ✅ 只算一次 predict_proba
proba_high = TabPFN_model.predict_proba(X_input)[0][1]
is_t = proba_high > sp
prob = (proba_high * 1000) // 1 / 10  # 保留你原来的“保留1位小数”的写法

result = 'High Risk Recurrence Group' if is_t else 'Low Risk Recurrence Group'

#%% UI action
if st.button('Predict'):
    st.markdown('## Result: ' + result)

    # ✅ Low Risk 出气球
    if result == 'Low Risk Recurrence Group':
        st.balloons()

    st.markdown('## Probability of High Risk Recurrence Group: ' + str(prob) + '%')
