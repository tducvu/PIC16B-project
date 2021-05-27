import tensorflow as tf 
import tensorflow.compat.v1 as tf
import keras
import numpy as np 
import os
import streamlit as st 
import PIL
from PIL import Image
import cv2
import time
from keras.models import load_model
import tensorflow.keras.backend as K 

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.applications.inception_resnet_v2 import preprocess_input
from skimage.transform import resize
from keras.applications.inception_resnet_v2 import InceptionResNetV2


st.markdown("<h1 style='text-align: center;'> Image Colorization</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'> Build with Tensorflow2 & Keras</h3>", unsafe_allow_html=True)

st.text('Upload grayscale images.')



# st.sidebar.title('1. Choose an image from 10 gray images')
# choice = st.sidebar.number_input(label = 'Enter a value: ', min_value=1, value=1, step=1)



# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    gray = np.array(im)
    return gray


# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    gray = load_image(uploadFile)
    st.image(gray)
    st.write("Image Uploaded Successfully")
    
    start_analyze_file=st.button('Colorize')
    
    
    if start_analyze_file == True:

        with st.spinner(text = 'Colorizing...'):
            time.sleep(5)


        
        #Load weights
        inception = InceptionResNetV2(weights='imagenet', include_top=True)
        inception.graph = tf.get_default_graph()


        def create_inception_embedding(grayscaled_rgb):
            grayscaled_rgb_resized = []
            for i in grayscaled_rgb:
                i = resize(i, (299, 299, 3), mode='constant')
                grayscaled_rgb_resized.append(i)
            grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
            grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
            with inception.graph.as_default():
                embed = inception.predict(grayscaled_rgb_resized)
            return embed

        st.cache(allow_output_mutation = True)

        color_me = img_to_array(gray)
        color_me = np.array(color_me, dtype=float)
        gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
        color_me_embed = create_inception_embedding(gray_me)
        color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
        color_me = color_me.reshape(color_me.shape+(1,))
        # gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
        
        # color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
        # color_me = color_me.reshape(color_me.shape+(1,)) 

        
        model = tf.keras.models.load_model("C:\\Users\\Alice\\Documents\\GitHub\\PIC16B-project\\model.hdf5")
        model.run_eagerly = False
        model.call = tf.function(model.call)
        
        output = model.predict([color_me, color_me_embed])
        output = output * 128

        # Output colorizations
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me[:,:,0]
        cur[:,:,1:] = output
        
        
        st.image(lab2rgb(cur).astype('uint8'), clamp=True)
        st.success('DONE!')

# else:
#     st.write("Make sure you image is in JPG/PNG Format.")




