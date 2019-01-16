from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras.applications.inception_v3 import preprocess_input
from keras import applications
from keras.optimizers import SGD, Adam, RMSprop, Nadam
import csv
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


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
            labels.append(row[1])
            image_name.append(row[2])
            
    return labels, image_name



def calculate_auc_and_roc(predicted, real, plot=False):
    
    y_results, names = load_predictions(predicted)
    y_2test, names_test = load_labels(real)

    #y_results, names = gf.load_predictions('Inception_predictions.csv')
    #y_2test, names_test = gf.load_labels('Real_values_test.csv')
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
        #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
    
    return auc_keras
        


# ------------------------directories of the datasets -------------------------------

train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/training/'
validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/validation/'
#test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/other_test_cases/case_4/rgb/'
test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/test_dataset_classes/'

# ---------------------- test with cat and dogs ------------------------------

#train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/training/'
#validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/validation/'
#test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/test_with_folders/'


# ---------------- load a base model --------------------------

ROWS = 139
COLS = 139

# --------------------- Image Data Generator-------------------
train_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)

# ------generators to feed the model----------------

train_gen = train_idg.flow_from_directory(train_data_dir,
                                          target_size=(ROWS, COLS),
                                          batch_size = 100)

validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                          target_size=(ROWS, COLS),
                                          batch_size = 100)
                                          
test_gen = test_idg.flow_from_directory(test_data_dir, 
                                        target_size=(ROWS, COLS), 
                                        shuffle=False,
                                        batch_size = 270)          
                                        
# -------------- Load the pretrained model--------------------

input_shape = (ROWS, COLS, 3)
nclass = len(train_gen.class_indices)
base_model = applications.InceptionV3(weights='imagenet', 
                                include_top=False, 
                                input_shape=(ROWS, COLS,3))
base_model.trainable = False
add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.1))
add_model.add(Dense(1024, activation='relu'))

add_model.add(Dense(nclass, activation='softmax'))
#add_model.add(Dense(1, activation='softmax'))


adam = Adam(lr=0.001)
sgd = SGD(lr = 0.001, momentum = 0.9)

model = add_model
model.compile(loss='categorical_crossentropy', 
              optimizer=adam,
              metrics=['accuracy'])
model.summary()


  
                                          
history = model.fit_generator(train_gen, 
                              epochs = 5, 
                              shuffle=1,
                              steps_per_epoch = 100,
                              validation_steps = 100,
                              validation_data = validation_gen, 
                              verbose=1)

#file_path="weights.best.hdf5"
#model.load_weights(file_path)
validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                          target_size=(ROWS, COLS),
                                          batch_size = 100)

evaluation = model.evaluate_generator(validation_gen, verbose = True, steps=10)
print(evaluation)

evaluation_0 = model.evaluate_generator(test_gen, verbose = True, steps=1)
print(evaluation_0)

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
df.to_csv("predictions_test_keras2.csv", index=False)


# --------------------more predictions--------------------------

                                          
val_2_gen = test_idg.flow_from_directory(validation_data_dir, 
                                        target_size=(ROWS, COLS), 
                                        shuffle=False,
                                        batch_size = 2842)          
                                        

predict2 = model.predict_generator(val_2_gen, verbose = True, steps=1)

#print(len(predicts))
#print(predicts[:270])
#print('second part')
#print(predicts[270:])
x_0 = [x[0] for x in predict2]
x_1 = [x[1] for x in predict2]
names = [os.path.basename(x) for x in val_2_gen.filenames]
print(len(x_0), len(names))

predict2 = np.argmax(predict2, axis=1)
label_index = {v: k for k,v in val_2_gen.class_indices.items()}
predicts2 = [label_index[p] for p in predict2]

df = pd.DataFrame(columns=['class_1', 'class_2', 'fname', 'over all'])
df['fname'] = [os.path.basename(x) for x in val_2_gen.filenames]
df['class_1'] = x_0
df['class_2'] = x_1
df['over all'] = predicts2
df.to_csv("predictions_val_keras2.csv", index=False)

# -----------now lets calculate the AUC---------------------------------

real_test = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_test.csv'
to_test = 'predictions_test_keras2.csv'
auch_0 = calculate_auc_and_roc(to_test, real_test)
print(auch_0)

real_val = '/home/william/m18_jorge/Desktop/THESIS/DATA/real_values/Real_values_validation_plus_Island.csv'
to_test_validation = 'predictions_val_keras2.csv'
auch_1 = calculate_auc_and_roc(to_test_validation, real_val)
print(auch_1)