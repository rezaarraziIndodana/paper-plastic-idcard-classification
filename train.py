import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping

import argparse
import cv2
import numpy as np
import os
import shutil
from PIL import Image
from glob import glob
import pydash as _

parser = argparse.ArgumentParser(description='classify paper id card anf plastic id card')
parser.add_argument('--train_dir', type=str, help='Directory of train set')
parser.add_argument('--test_dir', type=str, help='Directory of test set')

args = parser.parse_args()

INPUT_TRAIN_DIR = args.train_dir
INPUT_TEST_DIR = args.test_dir

def preprocess_data(X, Y):
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

if __name__ == "__main__":

    x_train, y_train = get_dataset(INPUT_TRAIN_DIR)
    x_test, y_test = get_dataset(INPUT_TEST_DIR)
    
    print((x_train.shape, y_train.shape))
    print((x_test.shape, y_test.shape))

    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    input_t = keras.Input(shape=(256, 256, 3))
    res_model = keras.applications.ResNet50(include_top=False,
                                        weights="imagenet",
                                        input_tensor=input_t)

    for layer in res_model.layers[:143]:
        layer.trainable = False
    # Check the freezed was done ok
    for i, layer in enumerate(res_model.layers):
        print(i, layer.name, "-", layer.trainable)

    to_res = (224, 224)

    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
    model.add(res_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2, activation='softmax'))

    check_point = keras.callbacks.ModelCheckpoint(filepath="model.h5",
                                              monitor="val_acc",
                                              mode="max",
                                              verbose=1,
                                              save_weights_only=False,
                                              save_best_only=True,
                                              )

    earlystopping = EarlyStopping(patience=10, verbose=1, monitor='val_acc', mode='max')

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=16, epochs=100,
                        validation_data=(x_test, y_test),
                        callbacks=[check_point, earlystopping])
    model.summary()
    
    model.save("model.h5")