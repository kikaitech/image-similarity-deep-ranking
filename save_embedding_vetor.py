# coding: utf-8

from keras.layers import Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from skimage import transform
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras import backend as K
import numpy as np
import argparse
import os
import cv2


def convnet_model_():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model


def deep_rank_model():

    convnet_model = convnet_model_()
    first_input = Input(shape=(224, 224, 3))
    first_conv = Conv2D(96, kernel_size=(8, 8), strides=(
        16, 16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3, 3), strides=(
        4, 4), padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=(224, 224, 3))
    second_conv = Conv2D(96, kernel_size=(8, 8), strides=(
        32, 32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7, 7), strides=(
        2, 2), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    final_model = Model(
        inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model


model = deep_rank_model()


model.load_weights('./deepranking.h5')

x_train = []
y_train = []
path = []

pathData = './dataset/'
for nameClass in os.listdir(pathData):
    pathImages = pathData + nameClass
    for nameImage in os.listdir(pathImages):
        pathImage = pathImages + "/" + nameImage
        print(pathImage)
        image2 = load_img(pathImage)
        image2 = img_to_array(image2).astype("float64")
        image2 = transform.resize(image2, (224, 224))
        image2 *= 1. / 255
        image2 = np.expand_dims(image2, axis=0)

        embedding2 = model.predict([image2, image2, image2])[0]

        x_train.append(embedding2)
        y_train.append(int(nameClass[4:]))
        path.append(pathImage)

np.savez_compressed('./All.npz',
                    x_train, y_train, path)
