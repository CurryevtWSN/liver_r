#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
from tabpfn import TabPFNClassifier
#from#from sklearn.metrics import confusion_matrix, ConfusionMatrixDi
#import#im
#os.#os.chdir(r"D:\data_analysis\machine_learning\肝癌肝移植\肝癌肝


#%%不提示warning信息
# st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Interpretable Prediction of Post-Transplant Recurrence in Hepatocellular Carcinoma Using TabPFN and SHAP')
st.title('Interpretable Prediction of Post-Transplant Recurrence in Hepatocellular Carcinoma Using TabPFN and SHAP')

#%%set variTabPFNles selection
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



#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
# st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Shi-Nan Wu, Xiamen University')
#传入数据
map = {'No':0,
       'Yes':1}

Portal_Hepatic_Vein_Tumor_Thrombus =map[Portal_Hepatic_Vein_Tumor_Thrombus]


# 数据读取，特征标注
#%%load model
TabPFN_model = joblib.load('TabPFN_model.pkl')

#%%load data
hp_train = pd.read_csv('train.csv')

target = ["Recurrence"]
y = np.array(hp_train[target])
sp = 0.5

is_t = (TabPFN_model.predict_proba(np.array([[Age,
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
            Preoperative_HBsAg]]))[0][1])> sp
prob = (TabPFN_model.predict_proba(np.array([[Age,
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
            Preoperative_HBsAg]]))[0][1])*1000//1/10
    

if st.button('Predict'):
    if is_t:
        result = 'High Risk Recurrence Group'
    else:
        result = 'Low Risk Recurrence Group'

    st.markdown('## Result: ' + result)

    # 低风险时放气球
    if result == 'Low Risk Recurrence Group':
        st.balloons()

    st.markdown(
        '## Probability of Recurrence Risk: ' + str(prob) + '%'
    )





