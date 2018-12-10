import numpy as np
import tensorflow as tf
import time
import os
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import TimeDistributed
from keras.layers import Lambda, concatenate
from keras.layers import LSTM, GRU, SimpleRNN, RNN

from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

inlin = False # True/False
#if inlin:
#    matplotlib inline
#else:
#    matplotlib

import matplotlib
import matplotlib.pyplot as plt

def loadImages(N=250):
    from scipy import misc
    def load_pics(folder,n):
        imgs = []
        for i in range(n):
            img = misc.imread(folder+"img_{:05}.png".format(i+1))
            ch = img[:,:,0]
            imgs.append(ch)
        return np.array(imgs)

    def load_labels(fn):
        return np.loadtxt(fn, usecols=0)

    base = os.getcwd()+"/images/"
    trainpic = load_pics(base+"imgTrn/", 1000)
    testpic = load_pics(base + "imgTst/", 1000)
    ntrain, width, height = trainpic.shape

    xtrain = (trainpic/np.float32(255)).reshape(1000, width, height, 1)
    xtest = (testpic/np.float32(255)).reshape(1000, width, height, 1)

    ytrain = load_labels(base+"Trn_trg.csv")
    ytest = load_labels(base+"Tst_trg.csv")

    xtrain = xtrain[:250]
    ytrain = ytrain[:250]
    
    return xtrain, ytrain, xtest, ytest, width, height

def loadMNIST():
    xtrain, ytrain, xtest, ytest = np.load("mnist.npy")
    width, height = xtrain.shape[1:3]
    return xtrain, ytrain, xtest, ytest, width, height

xtrain, ytrain, xtest, ytest, width, height = loadImages(10)
#plt.figure(1, figsize=(15,10))
#plt.imshow(xtrain[:10,:,:].reshape(10*width,height).T,cmap="gray")
#plt.axis("off")
#plt.show()

print(ytrain[:10])

def binary_pred_stats(ytrue, ypred, threshold=0.5):
    one_correct = np.sum((ytrue==1)*(ypred > threshold))
    zero_correct = np.sum((ytrue==0)*(ypred <= threshold))
    sensitivity = one_correct / np.sum(ytrue==1)
    specificity = zero_correct / np.sum(ytrue==0)
    accuracy = (one_correct + zero_correct) / len(ytrue)
    return sensitivity, specificity, accuracy

# Load the dataset, Rectangles and Circles
xtrain, ytrain, xtest, ytest, width, height = loadImages(250)

# Uncomment below to load parts of the MNIST database instead
# NOTE! When using MNIST, comment out third Conv2D/MaxPooling2D pair!
# xtrain, ytrain, xtest, ytest, width, height = loadMNIST()

# The size of the images
input_shape = (width, height, 1)

# Define the CNN model

#model = Sequential([
#    Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=input_shape),
#    MaxPooling2D(pool_size=(2, 2)),
#    Conv2D(8, kernel_size=(3, 3), activation='relu'),
#    MaxPooling2D(pool_size=(2, 2)),
#    Conv2D(8, kernel_size=(3, 3), activation='relu'),
#    MaxPooling2D(pool_size=(2, 2)),
#    Flatten(),
#    Dense(10, activation='relu'),
    #Dropout(0.5),
#   Dense(1),
#     Activation('sigmoid')
#])

### CNN suggested for ex 3 & 4.       
model = Sequential([  
    Conv2D(6, kernel_size=(3, 3),activation='relu',input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape),    
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(4, kernel_size=(3, 3),activation='relu',input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(5, activation='relu'),
    #Dropout(0.5),
    Dense(1),
    Activation('sigmoid')
])

# We use cross entropy error and the adam optimizer
adam = Adam(lr=0.005)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


# Now train the model

estimator = model.fit(xtrain, ytrain, 
                      epochs=30, 
                      batch_size=50,
                      verbose=0)

# Plot the training error
plt.plot(estimator.history['loss'])
plt.title('Model training')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(['train'], loc=0)
plt.show()

# Get the training predictions and results for those
predtrain = model.predict(xtrain)[:,0]
sensitivity, specificity, accuracy = binary_pred_stats(ytrain, predtrain)
print("train set:", sensitivity, specificity, accuracy)

# Get the test predictions and the results for those
predtest = model.predict(xtest)[:,0]
sensitivity, specificity, accuracy = binary_pred_stats(ytest, predtest)
print("test set: ", sensitivity, specificity, accuracy)

# if True then Maxpooling will be applied before showing the filter
post_pool = False

# The image index to show
idx = 11

kind = MaxPooling2D if post_pool else Conv2D
outs = [model.layers[0].input] + [l.output for l in model.layers if isinstance(l, kind)]
intermediate = K.function([model.layers[0].input, K.learning_phase()], outs)
print(ytest[idx])
print(sensitivity)
states = intermediate([xtest[idx:idx+1], 0])
plt.figure(figsize=(18,12))

for k,s in enumerate(states):
    plt.figure(figsize=(18,12))
    plt.subplot(len(outs),1,k+1)
    pics = s[0]
    pics = np.rollaxis(pics,2,0)
    rows = 2 if pics.shape[0] > 8 else 1
    cols = pics.shape[0]//rows
    imgshape = pics.shape[1:]
    pics = pics.reshape((rows,cols)+imgshape)
    pics = pics.swapaxes(1,2)
    pics = pics.reshape((pics.shape[0]*pics.shape[1], pics.shape[2]*pics.shape[3]))
    extent = (0,cols*imgshape[0], 0,rows*imgshape[1])
    plt.imshow(pics,cmap='gray',extent=extent)
    for r in range(1,rows):
        plt.plot([0,cols*imgshape[0]], [r*imgshape[1], r*imgshape[1]], color='r', linestyle='-', linewidth=1)
    for c in range(1,cols):
        plt.plot([c*imgshape[0], c*imgshape[0]], [0,rows*imgshape[1]], color='r', linestyle='-', linewidth=1)

plt.show()