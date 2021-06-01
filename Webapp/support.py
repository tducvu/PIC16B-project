# import necessary packages for CNN and image preprocessing
import numpy as np

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers.core import RepeatVector
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import (
    Conv2D,
    UpSampling2D,
    Input,
    Reshape,
    concatenate,
)

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

import tensorflow as tf


def create_inception_embedding(inception, grayscaled_rgb):
    # Create embedding
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant', anti_aliasing=True)
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


def load_pretrained_model(inception_wpath, colornet_wpath):
    """Load and return pre-trained model and the InceptionResNet model"""

    print('Loading pre-trained model...')

    # Load weights of InceptionResNet model for embedding extraction
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights(inception_wpath)
    inception.graph = tf.get_default_graph()

    def conv_stack(data, filters, s):
        """Utility for building conv layer"""
        output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
        return output

    embed_input = Input(shape=(1000,))

    # Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = conv_stack(encoder_input, 64, 2)
    encoder_output = conv_stack(encoder_output, 128, 1)
    encoder_output = conv_stack(encoder_output, 128, 2)
    encoder_output = conv_stack(encoder_output, 256, 1)
    encoder_output = conv_stack(encoder_output, 256, 2)
    encoder_output = conv_stack(encoder_output, 512, 1)
    encoder_output = conv_stack(encoder_output, 512, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)

    # Fusion
    # y_mid: (None, 256, 28, 28)
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output)

    # Decoder
    decoder_output = conv_stack(fusion_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = conv_stack(decoder_output, 16, 1)
    decoder_output = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

    # Load colornet weights
    model.load_weights(colornet_wpath)

    print('Model loaded!')
    return(model, inception)
