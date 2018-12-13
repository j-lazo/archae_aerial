import os
import random
import shutil


def generate_training_and_validation_sets(training_percentage=0.5):

    current_directory = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/'
    files_path_positives = "".join([current_directory, 'cats/'])
    files_path_negatives = "".join([current_directory, 'dogs/'])
    positive_images = os.listdir(files_path_positives)
    negative_images = os.listdir(files_path_negatives)

    training_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/training/'
    validation_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/cats_dogs/cats_and_dogs/validation/'

    for count_i, image in enumerate(positive_images):
        print(count_i,len(positive_images))
        if random.random() <= training_percentage:
            shutil.copy(files_path_positives + image, "".join([training_dir, 'cats/', image]))
        else:
            shutil.copy(files_path_positives + image, "".join([validation_dir, 'cats/', image]))
            
    for count_i, image in enumerate(negative_images):
        print(count_i,len(negative_images))
        if random.random() <= training_percentage:
            shutil.copy(files_path_negatives + image, "".join([training_dir, 'dogs/', image]))
        else:
            shutil.copy(files_path_negatives + image, "".join([validation_dir, 'dogs/', image]))


def main():
    generate_training_and_validation_sets(training_percentage=0.6)


if __name__ == '__main__':
    main()
