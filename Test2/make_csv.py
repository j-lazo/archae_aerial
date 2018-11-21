import os
import csv
import numpy as np
import random
from shutil import copyfile
directory1 = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/CODES/Test2/images/interesting'
directory2 = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/CODES/Test2/images/not_interesting'

sdt1 = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/CODES/Test2/images/imgTrn'
sdt2 = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/CODES/Test2/images/imgTst'

files1 = [f for f in os.listdir(directory1) if f[-4:] =='.jpg']
files2 = [f for f in os.listdir(directory2) if f[-4:] =='.jpg']
#all_files = files1 + files2
training = [] 
validation = []

for f in files1:
    if random.random() <= 0.75:
        training.append([1, 'imgTrn/' + f])
        copyfile("".join([directory1, '/', f]), "".join([sdt1, '/', 'img_', str(len(training)), '.jpg']))
    else:
        validation.append([1, 'imgTst/' + f])
        copyfile("".join([directory1, '/', f]), "".join([sdt2, '/', 'img_', str(len(validation)), '.jpg']))

for f in files2:
    if random.random() >= 0.75:
        training.append([0, 'imgTrn/' + f])
        copyfile("".join([directory2, '/', f]), "".join([sdt1, '/', 'img_', str(len(training)), '.jpg']))
    else:
        validation.append([0, 'imgTst' + f])
        copyfile("".join([directory2, '/', f]), "".join([sdt2, '/', 'img_', str(len(validation)), '.jpg']))

with open('images/Trn_trg.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for fi in training:
        filewriter.writerow(fi)


with open('images/Tst_trg.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    for fi in validation:
        filewriter.writerow(fi)

