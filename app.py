import streamlit as st
from fastai import *
from fastai.vision.all import *
from os import name
import pathlib

if name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
learn = load_learner(pathlib.Path('./model.pkl'))

st.title('Covid Classification Application')
st.write("This API is part of Chayoot Kosiwanich's project under the supervision of the AI Builders camp.")
st.write("Covid Classification Application was created to learn Deep Learning with Fastai Image Classification. And to extend my model's use to outsiders, I made this App publicly avalilable to use. If you're interested im my Image Classification Project, you can check it out here !")
st.text("https://kosiwanich.medium.com/covid-x-ray-classification-with-fastai-12cc1ae934c3")
st.title("Let's upload to get our prediction!")
st.write("file type support : .jpg .png .jpeg")

uploaded_file = st.file_uploader("Choose a file")

if (uploaded_file is not None) and (uploaded_file.name[-4:] in ['.jpg', '.png', 'jpeg']):
    st.image(uploaded_file, width=300)
    if st.button('Get the Prediction'):
        pred, pred_idx, probs = learn.predict(uploaded_file.getvalue())
        st.write(f"prediction : {pred} ｜ probability : {round(float(probs[pred_idx])*100, 2)}%")
        st.write("*** The results obtained from this App cannot be medical judged.")
elif (uploaded_file is not None): 
    st.write("Your uploaded image is not support. Try to uploaded others.  :D")