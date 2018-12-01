import os
import random
import shutil


def generate_training_and_validation_sets(training_percentage=0.1):

    current_directory = os.getcwd()
    files_path_positives = "".join([current_directory, '/', 'interesting/'])
    files_path_negatives = "".join([current_directory, '/', 'not_interesting/'])
    positive_images = os.listdir(files_path_positives)
    negative_images = os.listdir(files_path_negatives)

    training_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/training/'
    validation_dir = '/home/jl/MI_BIBLIOTECA/Escuela/Lund/IV/Thesis/test_data_set/training_vgg1/validation/'

    for count_i, image in enumerate(positive_images):
        if random.random() <= training_percentage:
            shutil.copy(files_path_positives + image, "".join([training_dir, 'positives/', image]))
        else:
            shutil.copy(files_path_positives + image, "".join([validation_dir, 'positives/', image]))

    for count_i, image in enumerate(negative_images):
        if random.random() <= training_percentage:
            shutil.copy(files_path_negatives + image, "".join([training_dir, 'negatives/', image]))
        else:
            shutil.copy(files_path_negatives + image, "".join([validation_dir, 'negatives/', image]))


def main():
    generate_training_and_validation_sets(training_percentage=0.15)


if __name__ == '__main__':
    main()
