from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump
import numpy as np
import os


def generate_dataset(folder_path):
    dataset = []
    labels = []
    for label_folder in os.listdir(folder_path):
        label_folder_path = os.path.join(folder_path, label_folder)
        if os.path.isdir(label_folder_path):
            label = int(label_folder.split('_')[0])
            for filename in os.listdir(label_folder_path):
                if filename.endswith(".npy"):
                    vector_path = os.path.join(label_folder_path, filename)
                    vector = np.load(vector_path)
                    dataset.append(vector)
                    labels.append(label)
    return dataset, labels


def train_svm(train_data, train_labels):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_labels)
    return clf


def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


def main():
    # Load your training data and labels
    train_folder_path = '/Users/omerhazan/Desktop/personal/studies/image ' \
                        'processing/image_processing_project/PCA/training_vectors_scored'
    test_folder_path = '/Users/omerhazan/Desktop/personal/studies/image ' \
                       'processing/image_processing_project/PCA/testing_vectors_scored'
    save_path = '/Users/omerhazan/Desktop/personal/studies/image ' \
                'processing/image_processing_project/PCA/trained_SVM_models'

    # Generate dataset for train, validation, and test
    train_data, train_labels = generate_dataset(train_folder_path)
    test_data, test_labels = generate_dataset(test_folder_path)
    # Train SVM model
    svm_model = train_svm(train_data, train_labels)

    # Evaluate model on test data
    test_accuracy = evaluate_model(svm_model, test_data, test_labels)
    print("Test Accuracy:", test_accuracy)

    model_name = f'svm_model_scored_{test_accuracy:.4f}.joblib'
    save_model_path = os.path.join(save_path, model_name)
    dump(svm_model, save_model_path)
    print(f"Model saved as '{save_model_path}'")


if __name__ == "__main__":
    main()
