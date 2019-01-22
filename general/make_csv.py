import os
import csv
import numpy as np

#directory_1 = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/training/'
#direcotry_2 = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/validation/'

directory_1 = '/home/william/m18_jorge/Desktop/THESIS/DATA/training_no_data_augment/training/'
direcotry_2 = '/home/william/m18_jorge/Desktop/THESIS/DATA/training_no_data_augment/validation/'

sub_dirs_1 = [f for f in os.listdir(directory_1)]
sub_dirs_2 = [f for f in os.listdir(direcotry_2)]

with open ('Real_values_training_plus_no_case4.csv', 'w') as csvfile:
    write = csv.writer(csvfile, delimiter = ',')
    for sub_dir in sub_dirs_1:
        images = [f for f in os.listdir(''.join([directory_1, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['0', '1', image])
            elif sub_dir == 'negatives':
                write.writerow(['1', '0',image])

with open ('Real_values_validation_plus_no_case4.csv', 'w') as csvfile:
    write = csv.writer(csvfile, delimiter = ',')
    for sub_dir in sub_dirs_2:
        images = [f for f in os.listdir(''.join([direcotry_2, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['0', '1', image])
            elif sub_dir == 'negatives':
                write.writerow(['1', '0', image])



with open ('Real_values_All_plus_no_case4.csv', 'w') as csvfile:
    write = csv.writer(csvfile, delimiter = ',')
    for sub_dir in sub_dirs_1:
        images = [f for f in os.listdir(''.join([directory_1, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['0', '1', image])
            elif sub_dir == 'negatives':
                write.writerow(['1', '0', image])
    
    for sub_dir in sub_dirs_2:
        images = [f for f in os.listdir(''.join([direcotry_2, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['0', '1', image])
            elif sub_dir == 'negatives':
                write.writerow(['1', '0', image])
     
        
    