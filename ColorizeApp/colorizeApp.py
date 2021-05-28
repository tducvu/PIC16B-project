# Streamlit App for the colorization
"""
Run the app by installing streamlit using pip
Type in cmd > streamlit run colorizeApp.py
"""

import streamlit as st
from load_css import local_scss
from model import colorize_model

import tensorflow as tf 
import keras
import numpy as np 
import os
import PIL
from PIL import Image
import cv2
import time


st.page_icon=":art:"
st.page_title="Image Colorization"



st.markdown("<h1 style='text-align: center;'>IMAGE COLORIZATION</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'> using TensorFlow</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'> PIC 16B Group Project</h3>", unsafe_allow_html=True)


####################################################
# choice = st.sidebar.number_input(label = 'Enter a value: ', min_value=1, value=1, step=1)
about = '''
### Abstract

>`Colorization` has a variety of applications in recreational and historical context. 
It transforms how we see and perceive photos which provides an immense value in helping us visualize and convey stories, emotion, as well as history. 
Our project scraped data using `selenium`
analyze and prepare the data, train them on a deep neural network deployed through `TensorFlow`,
and use `OpenCv` for image processing.

<div style='text-align: right;'>Alice Pham & Duc Vu</div>
'''

st.sidebar.markdown(about, unsafe_allow_html=True)

####################################################




md = '''
Start colorization with some Black & White images. You can upload an image or choose a pre-existed test images to try out the colorization
'''
st.markdown(md)

st.markdown("### Upload a B&W Picture:")
# Uploading File to Page: Choose your own GRAY picture to colorize
uploadFile = st.file_uploader(label="Upload Gray Image: ", type=['jpg', 'png', 'jpeg'])


# Checking the Format of the page
if uploadFile is not None:
    # Read and Load Image as np.array
    img = Image.open(uploadFile)
    gray = np.array(img)
    
    st.image(gray)
    
    start_analyze_file=st.button('Colorize')
    
    if start_analyze_file == True:

        with st.spinner(text = 'Colorizing...'):
            time.sleep(5)

        colorize_model(gray)

###################################################3

# If not upload, choose an image in our test images:
st.markdown("### Choose a B&W Picture from our small collection:")


img_folder = "C:\\Users\\Alice\\Documents\\GitHub\\PIC16B-project\\colorizer\\Test"

def load_img_from_folder(folder):
    imgs = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None: imgs.append(img)
    return imgs

images = load_img_from_folder(img_folder)
st.image([img for img in images])


i = st.number_input(label="Choose a test picture number:", min_value=1, value=1, step=1)
gray2 = images[i-1]

start_analyze_test_file=st.button('Colorize')

if start_analyze_test_file == True:
    colorize_model(gray2)



