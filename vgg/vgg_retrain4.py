import pandas as pd
import numpy as np
import os
import keras
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


train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation'


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
from keras.models import Model
custom_model = Model(input=vgg_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:7]:
    layer.trainable = False

# Do not forget to compile it
custom_model.compile(loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#included in our dependencies

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(100, 100),
                                                 color_mode='rgb',
                                                 batch_size=20,
                                                 class_mode='categorical',
                                                  shuffle=True)

step_size_train = train_generator.n//train_generator.batch_size


estimator = custom_model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=10)

print(estimator.__dict__.keys())
plt.figure()
plt.plot(estimator.history['acc'], label='train')
plt.plot(estimator.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

print(estimator.__dict__)
print(estimator.history.__dict__.keys())

plt.figure()
plt.plot(estimator.history['val_loss'], label='validation')
plt.title('Loss')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()