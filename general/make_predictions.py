from keras.layers import Dense
from keras.models import model_from_json
import general_functions as gf
import generate_map as gm
from keras.preprocessing.image import ImageDataGenerator
import datetime
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import regularizers
from keras.applications.mobilenet import preprocess_input
import numpy as np
import cv2
import os


def model():
    el2 = 0.01
    inception_base = InceptionV3(weights='imagenet', include_top=False)
    inception_base.summary()

    # add a global spatial average pooling layer
    x = inception_base.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(512, activation='relu')(x)

    # and a fully connected output/classification layer
    predictions = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(el2))(x)

    # create the full network so we can train on it
    inception_transfer = Model(input=inception_base.input, output=predictions)

    for layer in inception_base.layers:
        layer.trainable = False

    adam = Adam(lr=0.001)

    # Do not forget to compile it
    inception_transfer.compile(loss='categorical_crossentropy',
                               optimizer=adam,
                               metrics=['accuracy'])

    return inception_transfer


def main():

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    today = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d-%Hh%mm')
    loaded_model = model()
    # loaded_model = model_from_json()
    # load weights into new model

    file_weights = 'inception_weigths_20181213-18h12m_l2_0.01.h5'
    loaded_model.load_weights(file_weights)
    print("Loaded model from disk")

    file_predictions = 'results/Inception_predictions_20181213-18h12m_l2_0.01.csv'
    file_reals = 'real_values/Real_values_test.csv'
    # test_dataset = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/test_dont_touch/'
    test_dataset = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/Hjortahammar/All/'
    # test_dataset = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/Island/Ir_images/'

    picture_array, names = gf.load_pictures_1(test_dataset)
    labels, image_labels = gf.load_labels(file_reals)

    # evaluate loaded model on test data

    adam = Adam(lr=0.001)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    tests_results = loaded_model.predict(picture_array)
    print(np.shape(tests_results))
    y_2test = tests_results[:, 0]
    directory = test_dataset
    names_test = names
    list_of_images = os.listdir(test_dataset)

    print(tests_results[:, 0])

    whole_picture = gm.build_map(list_of_images, directory, 15, 18, [y_2test, names_test])
    cv2.imwrite('Hjortahammar_test.jpg', whole_picture)

    # score = loaded_model.evaluate(pictures_totest, results, verbose=1)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    main()

