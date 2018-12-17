import os
import random
import shutil


def generate_training_and_validation_sets(training_percentage=0.5):

    current_directory = '/home/william/m18_jorge/Desktop/THESIS/DATA/aerial_photos_plus/'
    files_path_positives = "".join([current_directory, 'positives/'])
    files_path_negatives = "".join([current_directory, 'negatives/'])
    positive_images = os.listdir(files_path_positives)
    negative_images = os.listdir(files_path_negatives)

    training_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/training/'
    validation_dir = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/validation/'
    
    all_training = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/all_training/'
    all_validation = '/home/william/m18_jorge/Desktop/THESIS/DATA/transfer_learning_training/all_validation/'

    for count_i, image in enumerate(positive_images):
        print(count_i,len(positive_images))
        if random.random() <= training_percentage:
            shutil.copy(files_path_positives + image, "".join([training_dir, 'positives/', image]))
            shutil.copy(files_path_positives + image, "".join([all_training, image]))
        else:
            shutil.copy(files_path_positives + image, "".join([validation_dir, 'positives/', image]))
            shutil.copy(files_path_positives + image, "".join([all_validation, image]))
            
    for count_i, image in enumerate(negative_images):
        print(count_i,len(negative_images))
        if random.random() <= training_percentage:
            shutil.copy(files_path_negatives + image, "".join([training_dir, 'negatives/', image]))
            shutil.copy(files_path_negatives + image, "".join([all_training, image]))
        else:
            shutil.copy(files_path_negatives + image, "".join([validation_dir, 'negatives/', image]))
            shutil.copy(files_path_negatives + image, "".join([all_validation, image]))


def main():
    generate_training_and_validation_sets(training_percentage=0.6)


if __name__ == '__main__':
    main()
