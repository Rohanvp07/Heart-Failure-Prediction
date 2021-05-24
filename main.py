
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open("model.pkl","rb"))

scaler = pickle.load(open("scaler.pkl","rb"))

 
page_bg_img = '''
<style>
body {
background-image: url("https://techcrunch.com/wp-content/uploads/2019/07/internet-heartbeat.gif?w=610");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
'''


st.markdown(
    """
<style>

.big-font {
    font-size:40px !important;
    color:blue;
    text-align:center;
}
.small-font {
    font-size:20px !important;
    color:white;
    text-align:center;

}
.error_class {
    font-size:25px !important;
    color:black;
    background:#ff6666;
    text-align:center;
    
}
.success_class {
    font-size:25px !important;
    color:black;
    background:#66ff66;
    text-align:center;
}
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-size:20px !important;
    font-family: monospace;
    
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.slider{
    color:black
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #ff5050;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}
.val{
    font-size:10px !important;
    color:black;
    text-align:center;
}
</style>
""",
    unsafe_allow_html=True,
)


def input_page():
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<p class="big-font"><b>Heart Failure Prediction</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Machine Learning Model</p>', unsafe_allow_html=True)
    

    gender = st.selectbox('Gender', ["Male","Female"])
    
    age = st.number_input('Age')

    highbp = st.selectbox('Do you have High Blood Pressure?', ["Yes","No"])    

    creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase')

    ejection_fraction = st.number_input('Ejection Fraction')

    platelets = st.number_input('Platelets')

    serum_creatinine = st.number_input('Serum Creatinine')
    
    serum_sodium = st.number_input('Serum Sodium')

    time = st.number_input('Time')
    

    btn = st.button("Predict")
    
    if btn:
        
        #age = scaler.transform(np.array(float(age)).reshape(-1,1))
        #creatinine_phosphokinase = scaler.transform(np.array(float(creatinine_phosphokinase)).reshape(-1,1))
        #ejection_fraction = scaler.transform(np.array(float(ejection_fraction)).reshape(-1,1))
        #platelets = scaler.transform(np.array(float(platelets)).reshape(-1,1))
        #serum_creatinine = scaler.transform(np.array(float(serum_creatinine)).reshape(-1,1))
        #serum_sodium = scaler.transform(np.array(float(serum_sodium)).reshape(-1,1))
        #time = scaler.transform(np.array(float(time)).reshape(-1,1))

        if gender=="Male":
            sex=1
        elif gender=="Female":
            sex=0

        if highbp=="Yes":
            highbp=1
        elif highbp=="No":
            highbp=0

            
        test=np.array([[age,creatinine_phosphokinase,ejection_fraction,highbp,platelets,
                        serum_creatinine,serum_sodium,sex,time]])
        result = model.predict(test)[0]
        if result==0:
            st.markdown('<p class="sucess_class"><b>Survived</b></p>', unsafe_allow_html=True)

        if result==1:
            st.markdown('<p class="error_class"><b>Dead</b></p>', unsafe_allow_html=True)

        

    
    
input_page()      
    
    
                  
