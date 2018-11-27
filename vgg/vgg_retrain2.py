import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier


# dimensions of our images.
img_width, img_height = 100, 100


train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation'
top_model_weights_path = 'vgg16_weights.h5'

#nb_train_samples = 8520
#nb_validation_samples = 1130
nb_train_samples = 100
nb_validation_samples = 100

epochs = 100
batch_size = 10

def load_labels(name):
    return np.loadtxt(name, usecols=0)

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


def train_top_model():

    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    y_pred_keras = model.predict(train_data).ravel()

    sensitivity, specificity, accuracy = binary_pred_stats(ytrain, y_pred_keras)

    print(len(y_pred_keras))
    print(len(validation_data))
    print(np.shape(validation_data))
    print(model.evaluate_generator(validation_data))
    #fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_data, y_pred_keras)
    #auc_keras = auc(fpr_keras, tpr_keras)
    #rf = RandomForestClassifier(max_depth=3, n_estimators=10)
    #rf.fit(train_data)


def main():
    save_bottlebeck_features()
    keras_model = train_top_model()


if __name__ == '__main__':
    main()