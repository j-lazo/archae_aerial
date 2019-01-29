# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:29:46 2019

@author: m18_jorge
"""

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
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate, Conv2D, Activation
from sklearn.metrics import roc_curve, auc
from keras import regularizers
from keras import backend as K


def load_labels(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(float(row[1]))
            image_name.append(row[2])
    return labels, image_name
    

def load_predictions(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row[1])
            image_name.append(row[2])
            
    return labels, image_name



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

def main(train_data_dir, validation_data_dir, test_data_dir_1, test_data_dir_2, idx=0, value=0.001, plot=False):
    
    img_width, img_height = 100, 100
    ROWS = img_width
    COLS = img_height
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    model = Sequential()
    model.add(Conv2D(512, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(256))
    #model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(value)))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
        
    adam = Adam(lr=0.0001)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    model.summary()
    #
    train_idg = ImageDataGenerator()
    val_idg = ImageDataGenerator()
    test_idg = ImageDataGenerator()
    test_idg2 = ImageDataGenerator()
    
    #train_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    #val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    #test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    #test_idg2 = ImageDataGenerator(preprocessing_function=preprocess_input)

    # ------generators to feed the model----------------

    train_gen = train_idg.flow_from_directory(train_data_dir,
                                      target_size=(ROWS, COLS),
                                      batch_size = 100)

    validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                      target_size=(ROWS, COLS),
                                      batch_size = 100)
                                      
    lenv_test1 = len(os.listdir(test_data_dir_1))                                     
    test_gen = test_idg.flow_from_directory(test_data_dir_1, 
                                    target_size=(ROWS, COLS), 
                                    shuffle=False,
                                    batch_size = 100)   
                                    
    test_gen2 = test_idg2.flow_from_directory(test_data_dir_2, 
                                    target_size=(ROWS, COLS), 
                                    shuffle=False,
                                    batch_size = 100)   

    # build the OwnCNN network
        
     # -----------here begins the important --------------------------
    
    history = model.fit_generator(train_gen, 
                epochs = 15, 
                shuffle=1,
                steps_per_epoch = 50,
                validation_steps = 50,
                validation_data = validation_gen, 
                verbose=1)

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

###-----------------------lets make predictions-------------------
    predicts = model.predict_generator(test_gen, verbose = True, steps=1)
    
    #print(len(predicts))
    #print(predicts[:270])
    #print('second part')
    #print(predicts[270:])
    x_0 = [x[0] for x in predicts]
    x_1 = [x[1] for x in predicts]
    names = [os.path.basename(x) for x in test_gen.filenames]
    print(len(x_0), len(names))
    
    predicts = np.argmax(predicts, 
                     axis=1)
    label_index = {v: k for k,v in train_gen.class_indices.items()}
    predicts = [label_index[p] for p in predicts]
    
    df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
    df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
    df['class_1'] = x_0
    df['class_2'] = x_1
    df['over all'] = predicts
    name_save_predictions_1 = ''.join(['predictions_rgb_OwnCNN_', str(idx), '_', str(value),  '_.csv'])   
    df.to_csv(name_save_predictions_1, index=False)
    
    
    
    
    # --------------------more predictions--------------------------
    len_val2 = len(os.listdir(test_data_dir_2))
    val_2_gen = test_idg.flow_from_directory(test_data_dir_2, 
                                        target_size=(ROWS, COLS), 
                                        shuffle=False,
                                        batch_size = len_val2)          
                                        
    
    predict2 = model.predict_generator(test_gen2, verbose = True, steps=1)
    
    #print(len(predicts))
    #print(predicts[:270])
    #print('second part')
    #print(predicts[270:])
    x_0 = [x[0] for x in predict2]
    x_1 = [x[1] for x in predict2]
    names = [os.path.basename(x) for x in test_gen2.filenames]
    print(len(x_0), len(names))
    
    predict2 = np.argmax(predict2, axis=1)
    label_index = {v: k for k,v in test_gen2.class_indices.items()}
    predicts2 = [label_index[p] for p in predict2]
    
    df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
    df['fname'] = [os.path.basename(x) for x in test_gen2.filenames]
    df['class_1'] = x_0
    df['class_2'] = x_1
    df['over all'] = predicts2
    name_save_predictions_2 = ''.join(['predictions_IR_OwnCNN_', str(idx), '_', str(value), '_.csv'])
    df.to_csv(name_save_predictions_2, index=False)
    
    # -------------------------predictions on the validation set --------------------------
    test_idg2 = ImageDataGenerator()
    va_gen2 = test_idg2.flow_from_directory(validation_data_dir, 
                                  target_size=(ROWS, COLS), 
                                   shuffle=False,
                                   batch_size = 100) 
                                   
    predict3 = model.predict_generator(va_gen2, verbose = True, steps=40)
    
    #print(len(predicts))
    #print(predicts[:270])
    #print('second part')
    #print(predicts[270:])
    x_0 = [x[0] for x in predict3]
    x_1 = [x[1] for x in predict3]
    names = [os.path.basename(x) for x in va_gen2.filenames[:4000]]
    print(len(x_0), len(names))
    
    predict3 = np.argmax(predict3, axis=1)
    label_index = {v: k for k,v in va_gen2.class_indices.items()}
    predict3 = [label_index[p] for p in predict3]
    
    df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
    df['fname'] = names
    df['class_1'] = x_0
    df['class_2'] = x_1
    df['over all'] = predict3
    name_save_predictions_3 = ''.join(['predictions_OwnCNN_', '_', str(idx), '_', str(value), '_.csv'])
    df.to_csv(name_save_predictions_3, index=False)
    
    # -----------now lets calculate the AUC---------------------------------
    
    real_test = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_case4_rgb.csv'
    auch_0 = calculate_auc_and_roc(name_save_predictions_1, real_test)
    print(auch_0, 'RGB')
    
    real_val = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_case4_IR.csv'
    auch_1 = calculate_auc_and_roc(name_save_predictions_2, real_val)
    print(auch_1, 'IR')
    
    real_val = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_validation_no_augment.csv'
    auch_2 = calculate_auc_and_roc(name_save_predictions_3, real_val)
    print(auch_2, 'validation data')
  



    # ----------------- save results ---------------------------

    today = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d-%Hh%mm')                          
    model.save_weights(''.join(['weights_incep_',today,'_dropout_',str(value),'_.h5']), True)

    with open(''.join(['Results_training', today,'_l2norm_', str(value), '_.csv']), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',' )
        writer.writerow(['Acc', 'Val_acc', 'Loss', 'Val_Loss'])
        for i, num in enumerate(history.history['acc']):
            writer.writerow([num, history.history['val_acc'][i], history.history['loss'][i], history.history['val_loss'][i]])

    if plot is True:
        plt.figure()
        """
        plt.plot([0, 1], [0, 1], 'k--')real_val
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

    #initial_dir = '/home/jl/aerial_photos_plus/'
    #folders = os.listdir(initial_dir)
    
    test_dir_rgb = '/home/william/m18_jorge/Desktop/THESIS/DATA/case4_test/rgb/'
    test_dir_ir = '/home/william/m18_jorge/Desktop/THESIS/DATA/case4_test/IR/'

    train_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/training/'
    val_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/validation/'

    #train_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/training_no_data_augment/training/'
    #val_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/training_no_data_augment/validation/'

    #train_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/training_no_data_augment/using_dgan_results/training/'
    #val_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/training_no_data_augment/using_dgan_results/validation/'

    indx = 0
    value = 0
    main(train_dir, val_dir, test_dir_rgb, test_dir_ir, indx)

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

