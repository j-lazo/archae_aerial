import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


base_model=MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top=False)
#imports the mobilenet model and discards the last 1000 neuron layer.

train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation'

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x) #dense layer 2
x = Dense(512, activation='relu')(x) #dense layer 3
#preds=Dense(120, activation='softmax')(x) #final layer with softmax activation
preds = Dense(2, activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)
# specify the inputs, amd the outputs

# now a model has been created based on our architecture
#for i, layer in enumerate(model.layers):
#    print(i,layer.name)

for layer in model.layers:
    layer.trainable = False

# or if we want to set the first 20 layers of the network to be non-trainable
"""for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True"""

adam = Adam(lr=0.001)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(128, 128),
                                                 color_mode='rgb',
                                                 batch_size=20,
                                                 class_mode='categorical',
                                                  shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                 target_size=(128, 128),
                                                 color_mode='rgb',
                                                 batch_size=20,
                                                 class_mode='categorical',
                                                  shuffle=True)


model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n//train_generator.batch_size

#print(train_generator.__dict__)
#print(train_generator.__dict__.keys())

model.summary()


nb_validation_samples = 100
batch_size = 20

estimator = model.fit_generator(generator=train_generator,
                                       steps_per_epoch=step_size_train,
                                       validation_data=validation_generator,
                                       validation_steps=nb_validation_samples // batch_size,
                                       epochs=15)
print(estimator.__dict__.keys())

plt.figure()
plt.plot(estimator.history['acc'], label='train')
plt.plot(estimator.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(estimator.history['loss'], label='train')
plt.plot(estimator.history['val_loss'], label='validation')
plt.title('Loss')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()