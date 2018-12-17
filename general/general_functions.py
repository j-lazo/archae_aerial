import csv
import cv2
import numpy as np
import os
from scipy import misc
from skimage import transform


def load_predictions(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row[1])

            image_name.append(row[0])
    return labels, image_name


def match_reals_and_prediction(file_reals, files_predictions, name_ouput):
    list_reals = []
    list_predictons = []
    common_names = []
    reals, name_reals = load_labels(file_reals)
    predictions, names_predictions = load_predictions(files_predictions)
    for i, name in enumerate(name_reals):
        for j, other_name in enumerate(names_predictions):
            if name == other_name:
                common_names.append(other_name)
                list_reals.append(float(reals[i]))
                list_predictons.append(float(predictions[j]))

    with open(''.join([name_ouput, '.csv']), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Name Picture', 'Real Value', 'Predicted Value'])
        for i, row in enumerate(common_names):
            writer.writerow(row, list_reals[i], list_predictons[i])


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
            image_name.append(row[1])
    return labels, image_name


def paint_image(image, value):
    im = cv2.imread(image)
    if value < 0.7:
        im[:, :, 1] = 255*value

    return im
