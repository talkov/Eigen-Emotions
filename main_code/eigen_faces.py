import cv2
import os
import numpy as np


def load_images_labels(directory, num_images=None, normalize=True):
    folder_name = directory.split("/")[-1]
    folder_name = folder_name.split("_")
    labels = folder_name[0]
    images_names = os.listdir(directory)
    images = []
    labels_encoded = []
    count = 0
    for img_name in images_names:
        if not is_image(img_name):
            continue
        img = cv2.imread(f'{directory}/{img_name}', cv2.IMREAD_GRAYSCALE)
        if normalize:
            img = img / 255
        images.append(img)
        labels_encoded.append(count)
        count += 1
        if num_images is not None and count >= num_images:
            break
    return np.array(images), np.array(labels_encoded)


def is_image(filename):
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    file_extension = filename.split('.')[-1].lower()
    return file_extension in image_extensions


def get_top_values_to_reach_threshold(sorted_values, threshold=0.95):
    total_sum = np.sum(sorted_values)
    target_sum = threshold * total_sum
    current_sum = 0
    selected_values = []

    for value in sorted_values:
        current_sum += value
        selected_values.append(value)
        if current_sum >= target_sum:
            break

    return selected_values


def create_emotions_matrices(data_paths, emotions, num_images=None, saving_path=None, normalize=True):
    # validate "emotions"
    for emo in emotions:
        if emo not in ['happy', 'sad', 'angry', 'surprised']:
            print('Sorry!\n this emotion is yet to be supported.\n you can file an emotion request in '
                  'www.fuckemotions.com and we will attend to your request as soon as possible')
            return False
    # set paths to ever emotion folder
    emotion_folders = [data_paths + '/{}_sorted_ofir_light'.format(emotion) for emotion in emotions]

    # Initialize dictionaries to hold models and weight matrices for each emotion
    models = {}
    weights_matrices = {}
    mean_matrices = {}

    # Load images and train models for each emotion
    for emotion, path in zip(emotions, emotion_folders):
        images, labels_encoded = load_images_labels(path, num_images, normalize)

        # Create and train a model for the current emotion
        model = cv2.face.EigenFaceRecognizer_create()
        model.train(images, labels_encoded)

        # Store the model in the dictionary
        models[emotion] = model

        # Retrieve and store the weight matrix for the current emotion
        weights_matrices[emotion] = model.getEigenVectors()
        mean_matrices[emotion] = model.getMean()
        if not num_images:
            images_in_matrix = len(images)
        else:
            images_in_matrix = num_images
        eigen_values = model.getEigenValues()
        top_eigen_values = get_top_values_to_reach_threshold(eigen_values)
        weights_matrices[emotion] = weights_matrices[emotion][:,0:len(top_eigen_values)]
        if saving_path:
            np.save(
                saving_path + f'/{emotion}_matrix',
                weights_matrices[emotion])
            np.save(
                saving_path + f'/{emotion}_mean_matrix',
                mean_matrices[emotion])

    return [weights_matrices, mean_matrices]
