import streamlit as st
import torch
from code.config import Config
from PIL import Image
config = Config()
import numpy as np
st.write(
'''
# Hello 👋 
# Namaste 🙏
# konichiwa 🙇

Here we generate anime images from real photos in styles of several anime creators like Makoto Shinkai, Hayao Miyazaki etc. If you are an anime fan boy or anime fan girl and also a fan of Machine Learning then you are at correct place.
We here use CycleGAN with identity loss for training and Generating images
'''
)

option = st.selectbox(
    label='\t',
    options =['select an option','camera','file upload']
    )
input_photo = None
match option:
    case 'select an option': st.markdown('#### I wonder, which method would you like to use to upload images 🤔?')
    case 'camera': 
        st.markdown('#### Say cheeseeeeeeee... ✌️')
        input_photo = st.camera_input(label='\t')
        # st.write(type(input_photo))
    case 'file upload': 
        st.markdown('Please Upload your image in png, jpeg, or jpg format 😊')
        input_photo = st.file_uploader(label='\t', type=['png','jpg','jpeg'])
        # st.write(type(input_photo))
            
if input_photo:
    # st.write(f'{type(input_photo)}')
    input_photo = Image.open(input_photo).convert('RGB')
    input_photo = config.preprocess(input_photo)
    # st.write(f'{type(input_photo)} {input_photo.shape}')
    st.write('the processed image taken as input is, please check how it looks 🥹🥹')
    st.session_state.input_photo = input_photo
    st.image((input_photo*0.5+0.5).permute(1,2,0).numpy(),clamp=True)