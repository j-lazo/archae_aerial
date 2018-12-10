from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
from scipy import misc
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skimage import transform
from keras.optimizers import SGD, Adam, RMSprop, Nadam


base = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/scripts/vgg/'

X, y = make_classification(n_samples=8590)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


def load_pictures(directory):

    lista1 = [f for f in os.listdir(directory + '/positives/')]
    lista2 = [f for f in os.listdir(directory + '/negatives/')]
    imgs = np.zeros([len(lista1) + len(lista2), 100, 100, 3])

    for i, image in enumerate(lista1):
        img = misc.imread(''.join([directory, '/positives/', image]))
        if np.array_equal(np.shape(img), (100, 100, 3)):
            imgs[i] = img
        else:
            img = transform.resize(img, (100, 100, 3))
            imgs[i] = img

    for i, image in enumerate(lista2):
        img = misc.imread(''.join([directory, '/negatives/', image]))
        if np.array_equal(np.shape(img), (100, 100, 3)):
            imgs[i + len(lista1)] = img
        else:
            img = transform.resize(img, (100, 100, 3))
            imgs[i + len(lista1)] = img


    array = np.array(imgs)
    array.reshape(len(imgs), 100, 100, 3)
    #return np.array(imgs[:])
    return array


def load_labels(csv_file):
    labels = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row[0])

    return labels


def binary_pred_stats(ytrue, ypred, threshold=0.5):
    one_correct = np.sum((ytrue==1)*(ypred > threshold))
    zero_correct = np.sum((ytrue==0)*(ypred <= threshold))
    sensitivity = one_correct / np.sum(ytrue==1)
    specificity = zero_correct / np.sum(ytrue==0)
    accuracy = (one_correct + zero_correct) / len(ytrue)
    return sensitivity, specificity, accuracy


def build_model():
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# dimensions of our images.
img_width, img_height = 100, 100
train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation'

#nb_train_samples = 8590
#nb_validation_samples = 1135
nb_train_samples = 2000
nb_validation_samples = 1000
epochs = 10
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

####### is this ok????????????????

model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

adam = Adam(lr=0.005)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

##################################

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print(dir(train_generator))

estimator = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


print('here we go... ')
print(type(validation_generator))
print(dir(validation_generator))


plt.plot(estimator.history['loss'], label='train')
plt.plot(estimator.history['val_loss'], label='validation')
plt.title('Loss')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(loc='best')

plt.figure()
plt.plot(estimator.history['acc'], label='train')
plt.plot(estimator.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(loc='best')

plt.show()


#--------------form here it is to predict the roc curve..... --------------------------

ytrain = load_labels(base + "imgTrn.csv")
y_test = load_labels(base + "imgVal.csv")

print(len(ytest), len(ytrain))
X_train = load_pictures(train_data_dir)
X_test = load_pictures(validation_data_dir)

y_pred_keras = model.predict(X_test).ravel()
print('predictions')
print(y_pred_keras)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras[:4295])

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

from sklearn.ensemble import RandomForestClassifier


# Supervised transformation based on random forests
#rf = RandomForestClassifier(max_depth=3, n_estimators=10)
#print(len(y_train))
#print(np.shape(X_train))
#rf.fit(X_train, y_train)


#y_pred_rf = rf.predict_proba(X_test)[:, 1]
#fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
#auc_rf = auc(fpr_rf, tpr_rf)


plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

# Zoom in view of the upper left corner.
plt.figure()
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()