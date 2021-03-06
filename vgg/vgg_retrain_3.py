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

train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/training/'
validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/validation/'


input_tensor = Input(shape=(100, 100, 3))
vgg_model = VGG16(weights='imagenet',
                               include_top=False,
                               input_tensor=input_tensor)

# To see the models' architecture and layer names, run the following
vgg_model.summary()

# Creating dictionary that maps layer names to the layers
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
x = layer_dict['block2_pool'].output

# Stacking a new simple convolutional network on top of it
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.

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

custom_model = Model(input=vgg_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:7]:
    layer.trainable = False

adam = Adam(lr=0.001)


custom_model.compile(loss='categorical_crossentropy',
                     optimizer=adam,
                     metrics=['accuracy'])


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#included in our dependencies

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(100, 100),
                                                 color_mode='rgb',
                                                 batch_size=100,
                                                 class_mode='categorical',
                                                  shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                 target_size=(100, 100),
                                                 color_mode='rgb',
                                                 batch_size=100,
                                                 class_mode='categorical',
                                                  shuffle=True)

step_size_train = train_generator.n//train_generator.batch_size


nb_validation_samples = 100
batch_size = 20


custom_model.save_weights('v_gg_weigths', True)


estimator = custom_model.fit_generator(generator=train_generator,
                                       steps_per_epoch=step_size_train,
                                       validation_data=validation_generator,
                                       validation_steps=nb_validation_samples // batch_size,
                                       epochs=15)

print(estimator.__dict__.keys())

with open ('results.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(['Acc', 'Val_Acc', 'Loss', 'Val_Loss'])
    for i, num in enumerate(estimator.history['acc']):
        writer.writerow([num, estimator.history['val_acc'][i], estimator.history['loss'][i], estimator.history['val_loss'][i]])


print(type(estimator.history['acc']))
print(estimator.history['acc'])
print(len(estimator.history['acc']))

plot = False
if plot is True:
    plt.figure()
    plt.plot(estimator.history['acc'], 'o-', label='train')
    plt.plot(estimator.history['val_acc'], 'o-', label='validation')
    plt.title('Accuracy')
    plt.ylabel('training error')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.plot(estimator.history['loss'], 'o-', label='train')
    plt.plot(estimator.history['val_loss'], 'o-', label='validation')
    plt.title('Loss')
    plt.ylabel('training error')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

