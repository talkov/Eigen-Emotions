import os
import numpy as np
from PIL import Image
from eigen_faces import create_emotions_matrices
import cv2

normelize = True


def flatten_image(image_path, is_normelize=normelize):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if is_normelize:
        img = img / 255
    return np.array(img).flatten()


def main():
    emotions = ["happy", "angry", 'surprised', 'sad']
    labels_dict = {'angry': 0, 'happy': 1, 'sad': 2, 'surprised': 3}
    # creating weight matrices
    create = input('do you want to create the matrices or load them c/l')
    if create[0].lower() == 'c':
        [weight_matrices, mean_matrices] = create_emotions_matrices(
            data_paths='/Users/omerhazan/Desktop/personal/studies/image '
                       'processing/image_processing_project/archive/train', emotions=emotions,
            saving_path='/Users/omerhazan/Desktop/personal/studies/image '
                        'processing/image_processing_project/PCA/eigen_matrices_95',num_images=None)
    # loading weight matrices if you dont want to create them
    else:
        weight_matrices = {}
        mean_matrices = {}
        weight_matrices['angry'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                           'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                           '/angry_matrix.npy')
        mean_matrices['angry'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                         'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                         '/angry_mean_matrix.npy')
        weight_matrices['happy'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                           'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                           '/happy_matrix.npy')
        mean_matrices['happy'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                         'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                         '/happy_mean_matrix.npy')
        weight_matrices['sad'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                         'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                         '/sad_matrix.npy')
        mean_matrices['sad'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                       'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                       '/sad_mean_matrix.npy')
        weight_matrices['surprised'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                               'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                               '/surprised_matrix.npy')
        mean_matrices['surprised'] = np.load('/Users/omerhazan/Desktop/personal/studies/image '
                                             'processing/image_processing_project/PCA/eigen_matrices/scored_eigen_matrices'
                                             '/surprised_mean_matrix.npy')
    for emotion in emotions:
        folder_path = f'/Users/omerhazan/Desktop/personal/studies/image ' \
                      f'processing/image_processing_project/archive/train/{emotion}_sorted_hist_eq_mean_0.5'
        output_folder = f'/Users/omerhazan/Desktop/personal/studies/image ' \
                        f'processing/image_processing_project/PCA/training_vectors_scored/{str(labels_dict[emotion])}_{emotion}_faces'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(folder_path):
            vector_list = []
            for emotion_internal in emotions:
                if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    image_path = os.path.join(folder_path, filename)
                    flattened_vector = flatten_image(image_path)
                    mean_matrix = mean_matrices[emotion_internal]
                    flattened_vector = flattened_vector - mean_matrix
                    emotion_matrix = weight_matrices[emotion_internal]
                    result_vector_emotion = np.dot(flattened_vector, emotion_matrix)
                    vector_list.append(np.resize(result_vector_emotion,[np.size(result_vector_emotion,1)]))
            if not vector_list:
                continue
            result_filename =os.path.splitext(filename)[0] + ".npy"
            result_path = os.path.join(output_folder, result_filename)
            result_vector = np.hstack(vector_list)
            np.save(result_path, result_vector)


if __name__ == "__main__":
    main()
