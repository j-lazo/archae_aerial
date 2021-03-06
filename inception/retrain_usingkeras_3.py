from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras.applications.inception_v3 import preprocess_input
from keras import applications
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras import regularizers
import csv
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from keras.models import model_from_json
import datetime

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

value = 0.008
idx = 00

train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/training/'
validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/validation/'
#test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/other_test_cases/case_4/rgb/'
test_data_dir_1 = '/home/william/m18_jorge/Desktop/THESIS/DATA/case4_test/rgb/'
test_data_dir_2 = '/home/william/m18_jorge/Desktop/THESIS/DATA/case4_test/IR/'

# ---------------------- test with cat and dogs ------------------------------

#train_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/training/'
#validation_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/validation/'
#test_data_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cacalculate_auc_and_rocts_dogs/cats_and_dogs/test_with_folders/'


# ---------------- load a base model --------------------------

ROWS = 139
COLS = 139
#name_val_dir = validation_data_dir[-3:]

# --------------------- Image Data Generator-------------------
train_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
val_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
test_idg2 = ImageDataGenerator(preprocessing_function=preprocess_input)

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
                                    batch_size = 200)   
                                    
lenv_test2 = len(os.listdir(test_data_dir_2))   
test_gen2 = test_idg2.flow_from_directory(test_data_dir_2, 
                                    target_size=(ROWS, COLS), 
                                    shuffle=False,
                                    batch_size = 200)   
                                    
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
#add_model.add(Dropout(value))
add_model.add(Dense(1024, activation='relu'))
#add_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(value)))

add_model.add(Dense(nclass, activation='softmax'))
#add_model.add(Dense(1, activation='softmax'))


adam = Adam(lr=0.0001)
sgd = SGD(lr = 0.0001, momentum = 0.9)

model = add_model
model.compile(loss='categorical_crossentropy', 
          optimizer='rmsprop',
          metrics=['accuracy'])
model.summary()

model.fit_generator(train_gen, 
                    epochs = 5, 
                    shuffle=1,
                    steps_per_epoch = 50,
                    validation_steps = 50,
                    validation_data = validation_gen, 
                    verbose=1)

for layer in model.layers[:249]:
    layer.trainable = False

for layer in model.layers[249:]:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', 
          optimizer=sgd,
          metrics=['accuracy'])

history = model.fit_generator(train_gen, 
                    epochs = 50, 
                    shuffle=1,
                    steps_per_epoch = 100,
                    validation_steps = 50,
                    validation_data = validation_gen, 
                    verbose=1)


today = datetime.datetime.strftime(datetime.datetime.today(), '%Y%m%d-%Hh%mm')                          
model.save_weights(''.join(['weights_incep_',today,'_dropout_',str(value),'_.h5']), True)

with open(''.join(['Results_training', today,'_l2norm_', str(value), '_.csv']), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',' )
    writer.writerow(['Acc', 'Val_acc', 'Loss', 'Val_Loss'])
    for i, num in enumerate(history.history['acc']):
        writer.writerow([num, history.history['val_acc'][i], history.history['loss'][i], history.history['val_loss'][i]])

#file_path="weights.best.hdf5"
#model.load_weights(file_path)
validation_gen = val_idg.flow_from_directory(validation_data_dir,
                                      target_size=(ROWS, COLS),
                                      batch_size = 100)

evaluation = model.evaluate_generator(validation_gen, verbose = True, steps=10)
print(evaluation, 'validation data')

evaluation_0 = model.evaluate_generator(test_gen, verbose = True, steps=1)
print(evaluation_0, 'RGB data')

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
name_save_predictions_1 = ''.join(['predictions_rgb_keras3_', str(idx), '_', str(value),  '_.csv'])   
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
name_save_predictions_2 = ''.join(['predictions_IR_keras3_', str(idx), '_', str(value), '_.csv'])
df.to_csv(name_save_predictions_2, index=False)

# -------------------------predictions on the validation set --------------------------
test_idg2 = ImageDataGenerator(preprocessing_function=preprocess_input)
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
name_save_predictions_3 = ''.join(['predictions_keras3_', '_', str(idx), '_', str(value), '_.csv'])
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