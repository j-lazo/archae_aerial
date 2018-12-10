import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import pandas as pd
import numpy as np
import os
import keras
import csv
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from sklearn.metrics import roc_curve

from scipy import misc
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skimage import transform
from keras.optimizers import SGD, Adam, RMSprop, Nadam
import csv
import datetime

#train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/training/'
#validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/validation/'

train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/transfer_learning/training/'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/transfer_learning/validation/'


def load_labels(csv_file):
    labels = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(int(row[0]))

    return labels


def load_pictures_1(directory):
    directory = directory
    lista = [f for f in os.listdir(directory)]
    imgs = np.zeros([len(lista), 100, 100, 3])

    for i, image in enumerate(lista):
        img = misc.imread(''.join([directory, image]))
        if np.array_equal(np.shape(img), (100, 100, 3)):
            imgs[i] = img
        else:
            img = transform.resize(img, (100, 100, 3))
            imgs[i] = img

    array = np.array(imgs)
    array.reshape(len(imgs), 100, 100, 3)
    # return np.array(imgs[:])
    return array, lista


def load_pictures(directory):

    names = []
    lista1 = [f for f in os.listdir(directory + '/positives/')]
    lista2 = [f for f in os.listdir(directory + '/negatives/')]
    imgs = np.zeros([len(lista1) + len(lista2), 100, 100, 3])

    for i, image in enumerate(lista1):
        img = misc.imread(''.join([directory, '/positives/', image]))
        names.append(image)
        if np.array_equal(np.shape(img), (100, 100, 3)):
            imgs[i] = img
        else:
            img = transform.resize(img, (100, 100, 3))
            imgs[i] = img

    for i, image in enumerate(lista2):

        img = misc.imread(''.join([directory, '/negatives/', image]))
        names.append(image)

        if np.array_equal(np.shape(img), (100, 100, 3)):
            imgs[i + len(lista1)] = img
        else:
            img = transform.resize(img, (100, 100, 3))
            imgs[i + len(lista1)] = img


    array = np.array(imgs)
    array.reshape(len(imgs), 100, 100, 3)
    #return np.array(imgs[:])
    return array, names

#import inception with pre-trained weights. do not include fully #connected layers
inception_base = InceptionV3(weights='imagenet', include_top=False)
inception_base.summary()

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer
predictions = Dense(2, activation='softmax')(x)

# create the full network so we can train on it
inception_transfer = Model(input=inception_base.input, output=predictions)

for layer in inception_base.layers:
    layer.trainable = False
    
adam = Adam(lr=0.001)

# Do not forget to compile it
inception_transfer.compile(loss='categorical_crossentropy',
                     optimizer=adam,
                     metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#included in our dependencies

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(100, 100),
                                                 color_mode='rgb',
                                                 batch_size=400,
                                                 class_mode='categorical',
                                                  shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                 target_size=(100, 100),
                                                 color_mode='rgb',
                                                 batch_size=400,
                                                 class_mode='categorical',
                                                  shuffle=True)

step_size_train = train_generator.n//train_generator.batch_size


nb_validation_samples = 200
batch_size = 20


inception_transfer.save_weights('inception_weigths', True)
            

estimator = inception_transfer.fit_generator(generator=train_generator,
                                       steps_per_epoch=step_size_train,
                                       validation_data=validation_generator,
                                       validation_steps=nb_validation_samples // batch_size,
                                       epochs=1)

print(estimator.__dict__.keys())

with open(''.join(['Inception_results_', str(datetime.datetime.now()), '.csv']), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Acc', 'Val_Acc', 'Loss', 'Val_Loss'])
    for i, num in enumerate(estimator.history['acc']):
        writer.writerow([num, estimator.history['val_acc'][i], estimator.history['loss'][i], estimator.history['val_loss'][i]])

test_dataset = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/transfer_learning/test/'
X_test, name_images_test = load_pictures_1(test_dataset)

tests_results = inception_transfer.predict(X_test)

with open(''.join(['Inception_predictions_', str(datetime.datetime.now()), '.csv']), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Name', 'Class 1', 'Class 2'])
    for i, row in enumerate(tests_results):
        writer.writerow([name_images_test[i], row[0], row[1]])


