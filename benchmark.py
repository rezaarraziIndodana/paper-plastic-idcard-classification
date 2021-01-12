import argparse
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from glob import glob
import numpy as np
import json
import pydash as _

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping

import shutil
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(description='classify paper id card anf plastic id card')
parser.add_argument('--test_dir', type=str, help='Directory of test set')
parser.add_argument('--model', type=str, default='model.h5', help='Path to .h5 model file')
parser.add_argument('--threshold', type=float, default=0.9, help='Threshold of MeanIOU for debugging')

args = parser.parse_args()

INPUT_TEST_DIR = args.test_dir
MODEL_FILE = args.model
THRESHOLD = args.threshold

def preprocess_data(X, Y):
    """
    a function that trains a convolutional neural network to classify the
    CIFAR 10 dataset
    :param X: X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    :return: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = keras.applications.resnet50.preprocess_input(X)
    Y_p = keras.utils.to_categorical(Y, 2)
    return X_p, Y_p

def read_image(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (256, 256))
    return image

def get_dataset(dir_):
    class_dir = sorted(os.listdir(dir_))
    print('class = ', {'0':class_dir[0], '1':class_dir[1]})
    x_ = []
    y_ = []

    for i, class_ in enumerate(class_dir):
        img_list = sorted(glob(dir_ + class_ + '/*'))
        for img in img_list:
            image = read_image(img)
            x_.append(image)
            y_.append([i])

    return np.asarray(x_), np.asarray(y_)

def evaluate_model(model, x, y):
    print(model.metrics_names)
    mean_iou = []
    evaluation = model.evaluate(x, y, batch_size=16, verbose=0)

    return evaluation

def main():
    if not os.path.exists(INPUT_TEST_DIR):
        print('Input directory not found ', INPUT_TEST_DIR)
    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)

        else:
            print('Load model... ', MODEL_FILE)
            to_res = (224, 224)
            model = load_model(MODEL_FILE, custom_objects={"tf": tf, "to_res":to_res})

            class_name = sorted(os.listdir(INPUT_TEST_DIR))
            X_test, Y_test = get_dataset(INPUT_TEST_DIR)
            x_test, y_test = preprocess_data(X_test, Y_test)

            print('Evaluation...')
            evaluation = evaluate_model(model, x_test, y_test)
            print('test loss, test acc: ',evaluation)

            #Confution Matrix and Classification Report
            Y_pred = model.predict(x_test)
            y_pred = np.argmax(Y_pred, axis=1)
            
            print('Confusion Matrix')
            print(confusion_matrix(Y_test, y_pred))
            print('Classification Report')
            print(classification_report(Y_test, y_pred, target_names=class_name))

            print('Done.')


if __name__ == '__main__':
    main()
