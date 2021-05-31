# import some needed libraries
import os
import numpy as np

from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

import matplotlib.pyplot as plt


# Create embedding
def create_inception_embedding(inception, grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant', anti_aliasing=True)
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


# Use matplotlib to show the result image 
def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def color_result(PATH, START, END, RESULT, model, inception):
    """
    FUNCTION
    --------
    Predict the colorized version of validation images
    
    INPUT
    PATH (str)      : directory path of the image folder needed to be evaluated on
    START, END (int): number/index of starting and ending image
    RESULT (?)      : resulting images
    model           : pre-trained tensorflow & keras model
    inception       : a classifier that trained on 1.2 mil images (InceptionResnet V2)
    """
    
    color_me = []
    i = 0
    # Take file in range [START, END] inside the PATH folder for evaluating
    for filename in os.listdir(PATH):
        if i > START and i < END:
            color_me.append(img_to_array(load_img(os.path.join(PATH, filename))))
        i += 1

    #Preprocessing from RGB to B&W and embedding
    color_me = np.array(color_me, dtype=float)
    color_me_embed = create_inception_embedding(inception, gray2rgb(rgb2gray(1.0/255*color_me)))
    color_me = rgb2lab(1.0/255*color_me)[:, :, :, 0]
    color_me = color_me.reshape(color_me.shape+(1,))

    # Test model
    output = model.predict([color_me, color_me_embed])
    # Rescale the output from [-1,1] to [-128, 128]
    output = output * 128

    # Create the result directory if not extists
    if not os.path.exists('Result'):
        os.makedirs('Result')

    # Output colorizations
    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        # LAB representation
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        # Save images as RGB
        imsave("Result/img_"+str(i)+".png", lab2rgb(cur))