import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

#import inception with pre-trained weights. do not include fully #connected layers
inception_base = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a fully connected output/classification layer
predictions = Dense(10, activation='softmax')(x)

# create the full network so we can train on it
inception_transfer = Model(input=inception_base.input, output=predictions)

for layer in inception_base.layers:
    layer.trainable = False
