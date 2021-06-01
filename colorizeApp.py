# This is the basic outline define the Streamlit App for the colorization
"""
To run the app, install streamlit first with either pip or conda.
Then, in the same folder with this file, run $ streamlit run colorizeApp.py
"""

import streamlit as st
from load_css import local_scss
from serving import load_model, evaluate_input
import numpy as np
import os
from io import BytesIO
from PIL import Image
from sys import platform

# UI of the app

# Title
st.page_title="Image Colorization"


st.markdown("<h1 style='text-align: center;'>Image Colorization</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> with TensorFlow & Keras</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'> PIC16B Group Project</h3>", unsafe_allow_html=True)


# Sidebar Abstract
about = '''
### Abstract

>`Colorization` has a variety of applications in recreational and historical context. 
It transforms how we see and perceive photos which provides immense value in helping us visualize and convey stories, emotion, as well as history. 
Our project was prepared and trained on approximately 10,000 images; specifically, we
implemented a convolutional neural network combined with a classifier deployed through `TensorFlow` and `Keras`.

<div style='text-align: right;'>Alice Pham & Duc Vu</div>
'''

st.sidebar.markdown(about, unsafe_allow_html=True)

####################################################



# App use
md = '''
Start colorization with some greyscale images. You can upload an image or choose a
pre-existed one from the below collection.
'''
st.markdown(md)

# Upload image
st.markdown("### Upload a B&W Picture:")
uploadFile = st.file_uploader(label="Upload Greyscale Image: ", type=['jpg', 'png', 'jpeg'])

if uploadFile is not None:
    # read and load image as a numpy array
    img = Image.open(uploadFile)
    gray = np.array(img)
    st.image(gray)

    start_analyze_file = st.button('Colorize', key='1')

    if start_analyze_file == True:
        with st.spinner(text = 'Colorizing...'):
            # load model, save image for processing, evaluate it and display the result
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

###################################################

# If not upload, choose an image in our test images:
st.markdown("### Choose an image:")

# Make sure path structure is appropriate for different OS
if platform == "Win32":
    img_folder = "colorizer\Test"
else:
    img_folder = "colorizer/Test"

def load_img_from_folder(folder):
    """
    Load images from a folder and return those images in a list
    and their corresponding path
    """
    imgs = []
    imgs_path = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None: imgs.append(img)
        imgs_path.append(os.path.join(folder,filename))
    return imgs_path, imgs
    
# Display images for testing
image_paths, images = load_img_from_folder(img_folder)
st.image([img for img in images])

i = st.text_input(label='Enter a tester number (from 1 to 8): ')
if i:
    try:
        i = int(i)
    except IndexError:
        print("Try an integer number from 1-8")

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
