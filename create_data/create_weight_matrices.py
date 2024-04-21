from eigen_faces import create_emotions_matrices
import numpy as np
import matplotlib.pyplot as plt

emotions = ['sad', 'angry', 'happy', 'surprised']
[weight_matrices, mean_matrices] = create_emotions_matrices(data_paths='/Users/omerhazan/Desktop/personal/studies'
                                                                       '/image '
                                                                       'processing/image_processing_project/archive'
                                                                       '/train',emotions=emotions,saving_path='/Users'
                                                                                                              '/omerhazan/Desktop/personal/studies/image processing/image_processing_project/PCA/eigen_matrices/tests2')
for emotion in emotions:
    weight_matrix = weight_matrices[emotion]
    for i in range(100):
        first_column_vector = weight_matrix[:, i]  # Extract the first column vector
        first_column_image = first_column_vector.reshape(48, 48)  # Reshape to 48x48

        # Display the resized first column vector
        plt.imshow(first_column_image, cmap='gray')
        plt.title(f"{i} column vector for {emotion}")
        plt.axis('off')
        plt.show()
