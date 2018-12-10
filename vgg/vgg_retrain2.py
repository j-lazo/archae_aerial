import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import backend as K
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from matplotlib import pyplot as plt
import csv
import os
from scipy import misc
from sklearn.metrics import roc_curve
from skimage import transform

from sklearn.datasets import make_classification

# dimensions of the images.
img_width, img_height = 100, 100

train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training'
#train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/aerial_photos_plus'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation'
top_model_weights_path = 'vgg16_weights.h5'
base = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/scripts/Test2/images/'


#nb_train_samples = 8520
#nb_validation_samples = 1130

nb_train_samples = 100
nb_validation_samples = 100
epochs = 10
batch_size = 10


def binary_pred_stats(ytrue, ypred, threshold=0.5):
    one_correct = np.sum((ytrue==1)*(ypred > threshold))
    zero_correct = np.sum((ytrue==0)*(ypred <= threshold))
    sensitivity = one_correct / np.sum(ytrue==1)
    specificity = zero_correct / np.sum(ytrue==0)
    accuracy = (one_correct + zero_correct) / len(ytrue)
    return sensitivity, specificity, accuracy


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
            img = transform.resize(img, (100, 100,3))
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


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=None)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(generator, int(nb_train_samples/10))
    generatore = open('bottleneck_fc_model.h5', mode='w')
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def build_model():

    # dimensions of our images.
    img_width, img_height = 100, 100

    train_data = np.load('bottleneck_features_train.npy')

    #print(np.shape(train_data[1:]), 'train data')
    #print(train_data[1])
    #print(train_data[10])
    #print(train_data.shape[:-1], 'train data[1]')

    #------ where the actual model is build--------------
    model = Sequential()
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    #model.add(Flatten(input_shape=train_data.shape[0:-1]))


    model.add(Flatten(input_shape=(100, 100, 3)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(lr=0.0005)
    rmsprop = 'rmsprop'
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_top_model():

    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    print(np.shape(train_data), 'shape_train_data')
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    ker_model = build_model()
    #print(np.shape(validation_data[0]))
    #print(np.shape(validation_labels))
    #print(np.shape(train_data))
    #print(np.shape(train_labels))
    fit_model = ker_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    plt.plot(fit_model.history['loss'], label='train')
    plt.plot(fit_model.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.ylabel('training error')
    plt.xlabel('epoch')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(fit_model.history['acc'], label='train')
    plt.plot(fit_model.history['val_acc'], label='validation')
    plt.title('Accuracy')
    plt.ylabel('training error')
    plt.xlabel('epoch')
    plt.legend(loc='best')

    return ker_model


def main():

    save_bottlebeck_features()
    keras_model = train_top_model()

    ytrain = load_labels(base + "imgTrn.csv")
    ytest = load_labels(base + "imgVal.csv")

    X_train = load_pictures(train_data_dir)
    X_test = load_pictures(validation_data_dir)

    print(type(X_test))
    print(np.shape(X_test))
    print(len(X_test))
    print(X_test[0])

    print()

    y_pred_keras = keras_model.predict(X_test).ravel()

    print(type(y_pred_keras))
    print(np.shape(y_pred_keras))
    print(len(y_pred_keras))
    print(y_pred_keras[0])


    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    from sklearn.metrics import auc
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.show()
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
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
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()