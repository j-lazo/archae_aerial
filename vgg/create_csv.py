import csv
import os


train_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training'
validation_data_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation'


files_train = os.listdir(train_data_dir)
positive_files_train = os.listdir(''.join([train_data_dir, '/', files_train[0]]))
negative_files_train = os.listdir(''.join([train_data_dir, '/', files_train[1]]))

files_validation = os.listdir(validation_data_dir)
positive_files_validation = os.listdir(''.join([validation_data_dir, '/', files_validation[0]]))
negative_files_validation = os.listdir(''.join([validation_data_dir, '/', files_validation[1]]))

with open('imgTrn.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for image in positive_files_train:
        writer.writerow(['1', image])

    for image in negative_files_train:
        writer.writerow(['0', image.strip('imgTrn/')])

with open('imgVal.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for image in positive_files_validation:
        writer.writerow(['1', image])

    for image in negative_files_validation:
        writer.writerow(['0', image])

