import argparse
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

import tensorflow as tf
import keras
from keras.models import load_model

parser = argparse.ArgumentParser(description='classify paper id card anf plastic id card')
parser.add_argument('input', type=str, help='Image (with IDCard) Input file')
parser.add_argument('--model', type=str, default='model.h5', help='Path to .h5 model file')

args = parser.parse_args()

INPUT_FILE = args.input
MODEL_FILE = args.model

class_label = {0:'ktp_normal', 1:'ktp_paper'}

def load_image():
    image = cv2.imread(INPUT_FILE)
    image = cv2.resize(image, (256, 256))
    return image

def predict_image(model, image):
    predict = model.predict(image, verbose=1)
    print(predict)
    return (predict[0][1]>0.5)

def main():
    if not os.path.isfile(INPUT_FILE):
        print('Input image not found ', INPUT_FILE)
    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)

        else:
            print('Load model... ', MODEL_FILE)

            to_res = (224, 224)
            model = load_model(MODEL_FILE, custom_objects={"tf": tf, "to_res":to_res})
            model.summary()
            
            print('Load image... ', INPUT_FILE)
            img = load_image()
            x = keras.applications.resnet50.preprocess_input(np.array([img]))

            print('Prediction...')
            prediction = predict_image(model, x)
            print(class_label[prediction])

            print('Done.')


if __name__ == '__main__':
    main()
