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
import sys
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="Path to the deep ranking model")

# ap.add_argument("-i1", "--image1", required=True,
#                 help="Path to the first image")

# ap.add_argument("-i2", "--image2", required=True,
#                 help="Path to the second image")

args = vars(ap.parse_args())

if not os.path.exists(args['model']):
    print("The model path doesn't exist!")
    exit()

# if not os.path.exists(args['image1']):
#     print("The image 1 path doesn't exist!")
#     exit()

# if not os.path.exists(args['image2']):
#     print("The image 2 path doesn't exist!")
#     exit()

# args = vars(ap.parse_args())


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

# for layer in model.layers:
#     print (layer.name, layer.output_shape)

model.load_weights(args['model'])
print('Load model done\nImage paths: ', end='')

stop_threads = False

def displayImg(path, img, orig):
    cv2.imshow(path, img)
    if orig == True:
        cv2.waitKey(0)
    else:
        while(stop_threads == False):
            cv2.waitKey(33)

for img_path in sys.stdin:
    try:
        img_path = img_path.strip()
        orig_thread = threading.Thread(target = displayImg, args = (img_path, cv2.imread(img_path), True))
        orig_thread.start()

        image1 = load_img(img_path)
        image1 = img_to_array(image1).astype("float64")
        image1 = transform.resize(image1, (224, 224))
        image1 *= 1. / 255
        image1 = np.expand_dims(image1, axis=0)

        distance = {}
        arr = []

        embedding1 = model.predict([image1, image1, image1])[0]
        data = np.load('./All.npz')
        x_train, y_train, path = data['arr_0'], data['arr_1'], data['arr_2']
        for i in range(0, len(x_train)):
            embedding2 = x_train[i]
            temp = sum([(embedding1[idx] - embedding2[idx]) **
                        2 for idx in range(len(embedding1))])**(0.5)
            pathImage = path[i]
            distance[pathImage] = temp
            arr.append(temp)

        arr.sort()
        threads = []
        stop_threads = False
        for key in distance:
            if abs(distance[key] - arr[0]) < 0.00001 or abs(distance[key] - arr[1]) < 0.00001 or abs(distance[key] - arr[2]) < 0.00001:
                t = threading.Thread(target = displayImg, args = (key, cv2.imread(key), False))
                threads.append(t)
                t.start()
        orig_thread.join()
        stop_threads = True
        for t in threads:
            t.join()
    except Exception as e:
        print(e)
        pass

# bestValue = 9999999.9
# bestImage = ""
# for key in distance:
#     if distance[key] < bestValue:
#         bestValue = distance[key]
#         bestImage = key
# print(bestImage)
# showimg = cv2.imread(bestImage)
# cv2.imshow("result", showimg)
# cv2.waitKey()

# image2 = load_img(args['image2'])
# image2 = img_to_array(image2).astype("float64")
# image2 = transform.resize(image2, (224, 224))
# image2 *= 1. / 255
# image2 = np.expand_dims(image2, axis=0)

# embedding2 = model.predict([image2, image2, image2])[0]

# distance = sum([(embedding1[idx] - embedding2[idx]) **
#                 2 for idx in range(len(embedding1))])**(0.5)
