import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD, Adam, RMSprop, Nadam
import csv
from matplotlib import pyplot as plt
import datetime
import os
import pandas as pd
import shutil


def copy_files(initial_dir, final_dir):
    subfolders_initial = os.listdir(initial_dir)
    subfolder_final = os.listdir(final_dir)
    for folder in subfolders_initial:
        image_list = os.listdir(initial_dir + folder)
        for image in image_list:
            file_name = ''.join([initial_dir, folder, '/', image])
            destination = ''.join([final_dir, folder, '/', image])
            print(file_name)
            shutil.copyfile(file_name, destination)


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


def load_labels(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(float(row[0]))
            image_name.append(row[2])
    return labels, image_name


def load_predictions(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row[0])
            image_name.append(row[2])

    return labels, image_name


def calculate_auc_and_roc(predicted, real, plot=False):
    y_results, names = load_predictions(predicted)
    y_2test, names_test = load_labels(real)

    # y_results, names = gf.load_predictions('Inception_predictions.csv')
    # y_2test, names_test = gf.load_labels('Real_values_test.csv')
    y_test = []
    y_pred = []

    print(len(y_results), len(names))
    print(len(y_2test), len(names_test))

    for i, name in enumerate(names):
        for j, other_name in enumerate(names_test):
            if name == other_name:
                y_pred.append(float(y_results[i]))
                y_test.append(int(y_2test[j]))

    print(len(y_pred))
    print(len(y_test))
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

    auc_keras = auc(fpr_keras, tpr_keras)

    if plot is True:
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

    return auc_keras


def main(train_data_dir, validation_data_dir, test_data_dir_1, test_data_dir_2, value=0.1, plot=False):
    # ------------------------directories of the datasets -------------------------------

    # train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/training/'
    # validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/validation/'
    # test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/other_test_cases/case_4/rgb/'
    # test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/test_dataset_classes/'

    # ---------------------- test with cat and dogs ------------------------------

    # train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/training/'
    # validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/validation/'
    # test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/test_with_folders/'

    # ---------------- load a base model --------------------------

    img_width, img_height = 150, 150
    top_model_weights_path = 'bottleneck_fc_model.h5'
    nb_train_samples = 1000
    nb_validation_samples = 1000
    epochs = 50
    batch_size = 16

    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    
    # build generators and build bottlenecks
    
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,
        class_mode=None)
    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(validation_generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

     # -----------here begins the important --------------------------

    #save_bottlebeck_features()
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    nclass = len(train_gen.class_indices)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclass, activation='softmax'))

    # optimizers

    adam = Adam(lr=0.001)
    sgd = SGD(lr=0.001, momentum=0.9)
    rms = 'rmsprop'
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    estimator = model.fit(train_data, train_labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)
    print(estimator.__dict__.keys())


    # --------------- evaluate the model -----------------

    val_idg = ImageDataGenerator()
    validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=100)

    evaluation = model.evaluate_generator(validation_gen, verbose=True, steps=10)
    print(evaluation, 'Validation dataset')

    test_idg = ImageDataGenerator()
    test_gen = test_idg.flow_from_directory(test_data_dir_1,
                                            target_size=(img_width, img_height),
                                            shuffle=False,
                                            batch_size = 200)

    evaluation_0 = model.evaluate_generator(test_gen, verbose=True, steps=1)
    print(evaluation_0, 'Test dataset RGB')

    test_idg2 = ImageDataGenerator()
    test_gen2 = test_idg2.flow_from_directory(test_data_dir_2,
                                              target_size=(img_width, img_height),
                                              shuffle=False,
                                              batch_size=200)

    evaluation_1 = model.evaluate_generator(test_gen2, verbose=True, steps=1)
    print(evaluation_1, 'Test dataset IR')

    # --------------- make predictions -------------------

    predicts_1 = model.predict_generator(test_gen, verbose=True, steps=1)
    predicts_2 = model.predict_generator(test_gen2, verbose=True, steps=1)

    # -------------------save predictions  ------------------------
    today = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d-%Hh%mm')
    x_0 = [x[0] for x in predicts_1]
    x_1 = [x[1] for x in predicts_1]
    names = [os.path.basename(x) for x in test_gen.filenames]
    print(len(x_0), len(names))

    predicts = np.argmax(predicts_1,
                         axis=1)
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    predicts = [label_index[p] for p in predicts]

    df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
    df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
    df['class_1'] = x_0
    df['class_2'] = x_1
    df['over all'] = predicts
    name_to_save_1 = ''.join(['predictions_vgg_case4_rgb_', today, '_.csv'])
    df.to_csv(name_to_save_1, index=False)

    # IR
    x_0 = [x[0] for x in predicts_2]
    x_1 = [x[1] for x in predicts_2]
    names = [os.path.basename(x) for x in test_gen2.filenames]
    print(len(x_0), len(names))

    predicts = np.argmax(predicts_2,
                         axis=1)
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    predicts = [label_index[p] for p in predicts]

    df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
    df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
    df['class_1'] = x_0
    df['class_2'] = x_1
    df['over all'] = predicts
    name_to_save_2 = ''.join(['predictions_vgg_case4_IR_', today, '_.csv'])
    df.to_csv(name_to_save_2, index=False)

    # calculate ROC and AUC

    real_test = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_case4_rgb.csv'
    to_test = name_to_save_1
    auch_0 = calculate_auc_and_roc(to_test, real_test)
    print(auch_0)

    real_val = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_case4_IR.csv'
    to_test_validation = name_to_save_2
    auch_1 = calculate_auc_and_roc(to_test_validation, real_val)
    print(auch_1)



    # ----------------- save results ---------------------------

    with open(''.join(['vgg_results_', today, '_.csv']), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Acc', 'Val_Acc', 'Loss', 'Val_Loss'])
        for i, num in enumerate(estimator.history['acc']):
            writer.writerow(
                [num, estimator.history['val_acc'][i], estimator.history['loss'][i], estimator.history['val_loss'][i]])

    if plot is True:
        plt.figure()
        """
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
        plt.show()"""


if __name__ == "__main__":

    initial_dir = '/home/jl/aerial_photos_plus/'
    folders = os.listdir(initial_dir)
    test_dir_rgb = '/home/william/m18_jorge/Desktop/THESIS/DATA/case4_test/rgb/'
    test_dir_ir = '/home/william/m18_jorge/Desktop/THESIS/DATA/case4_test/IR/'

    train_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/transfer_learning/training/'
    val_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/transfer_learning/validation/'

    main(train_dir, val_dir, test_dir_rgb, test_dir_ir)


    """posible_values = [0.1, 0.25, 0.5, 0.75]
    train_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/tem_train/'
    if (os.path.isdir(train_dir)):
        shutil.rmtree(train_dir)
    print(folders)
    number_folders = list(np.arange(0, len(folders), 1))

    for num, subfolder in enumerate(folders):
        number_folders.remove(number_folders.index(num))
        val_dir = ''.join([initial_dir, subfolder])
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        positives_dir = ''.join([train_dir, 'positives'])
        negatives_dir = ''.join([train_dir, 'negatives'])
        if not os.path.isdir(positives_dir):
            os.mkdir(positives_dir)
        if not os.path.isdir(negatives_dir):
            os.mkdir(negatives_dir)

        for remaining in number_folders:
            print(folders[remaining])
            check_folder = ''.join([initial_dir, folders[remaining], '/'])
            copy_files(check_folder, train_dir)

        main(train_dir, val_dir, test_dir_rgb, test_dir_ir, posible_values[num])
        shutil.rmtree(train_dir)
        number_folders = list(np.arange(0, len(folders), 1))"""

