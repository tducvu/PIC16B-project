# Streamlit App for the colorization
"""
Run the app by installing streamlit using pip
Type in cmd > streamlit run colorizeApp.py
"""

import streamlit as st
from load_css import local_scss
from serving import (
        load_model,
        evaluate_input,
)
import numpy as np
import os
from io import BytesIO
from PIL import Image
import cv2
from sys import platform

# basic UI of the app
st.page_title="Image Colorization"


st.markdown("<h1 style='text-align: center;'>Image Colorization</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> with TensorFlow</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'> PIC16B Group Project</h3>", unsafe_allow_html=True)


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

    start_analyze_file = st.button('Colorize', key='1')

    if start_analyze_file == True:
        with st.spinner(text = 'Colorizing...'):
            load_model()
            input_buffer = BytesIO()
            output_buffer = BytesIO()
            img.save(input_buffer, 'JPEG')
            input_img = evaluate_input(input_buffer)
            input_img.save(output_buffer, format='JPEG')
            output_img = Image.open(output_buffer)
            color = np.array(output_img)
            st.image(color)
            st.success("Done!")

###################################################3

# If not upload, choose an image in our test images:
st.markdown("### Choose a B&W Picture from our small collection:")


if platform == "Win32":
    img_folder = "colorizer\Test"
else:
    img_folder = "colorizer/Test"

def load_img_from_folder(folder):
    imgs = []
    imgs_path = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None: imgs.append(img)
        imgs_path.append(os.path.join(folder,filename))
    return imgs_path, imgs
    

image_paths, images = load_img_from_folder(img_folder)
st.image([img for img in images])

# i = st.number_input(label="Choose a test picture number:", min_value=1, value=1, step=1)
i = st.text_input(label='Enter a tester number (from 1 to 10): ')
if i:
    try: i = int(i)
    except ValueError: print("Try an integer number from 1-10")

    img2 = Image.open(image_paths[i-1])
    st.image(np.array(img2))

    start_analyze_test_file = st.button('Test Colorize', key='2')

    if start_analyze_test_file == True:
        with st.spinner(text = 'Colorizing...'):
            load_model()
            input_buffer = BytesIO()
            output_buffer = BytesIO()
            img2.save(input_buffer, 'JPEG')
            input_img = evaluate_input(input_buffer)
            input_img.save(output_buffer, format='JPEG')
            output_img = Image.open(output_buffer)
            color = np.array(output_img)
            st.image(color)
            st.success("Done!")
