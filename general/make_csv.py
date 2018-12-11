import os
import csv
import numpy as np

directory_1 = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/training/'
direcotry_2 = '/home/william/m18_jorge/Desktop/THESIS/DATA/trasnfer_learning_training/validation/'

sub_dirs_1 = [f for f in os.listdir(directory_1)]
sub_dirs_2 = [f for f in os.listdir(direcotry_2)]

with open ('Real_values_training.csv', 'w') as csvfile:
    write = csv.writer(csvfile, delimiter = ',')
    for sub_dir in sub_dirs_1:
        images = [f for f in os.listdir(''.join([directory_1, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['1', image])
            elif sub_dir == 'negatives':
                write.writerow(['0', image])

with open ('Real_values_validation.csv', 'w') as csvfile:
    write = csv.writer(csvfile, delimiter = ',')
    for sub_dir in sub_dirs_2:
        images = [f for f in os.listdir(''.join([direcotry_2, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['1', image])
            elif sub_dir == 'negatives':
                write.writerow(['0', image])



with open ('Real_values_All.csv', 'w') as csvfile:
    write = csv.writer(csvfile, delimiter = ',')
    for sub_dir in sub_dirs_1:
        images = [f for f in os.listdir(''.join([directory_1, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['1', image])
            elif sub_dir == 'negatives':
                write.writerow(['0', image])
    
    for sub_dir in sub_dirs_2:
        images = [f for f in os.listdir(''.join([direcotry_2, '/', sub_dir]))]
        for image in images:
            if sub_dir == 'positives':
                write.writerow(['1', image])
            elif sub_dir == 'negatives':
                write.writerow(['0', image])
     
        
    