import os
import cv2
import numpy as np


def lightened_img(img, mult):  # sets the average of the pixels to be 0.5 (to make sure that the light is not extreme)
    rows = np.shape(img)[0]
    cols = np.shape(img)[1]
    for row in range(rows):
        for col in range(cols):
            if img[row, col] * mult < 1:
                img[row, col] = img[row, col] * mult * 255
            else:
                img[row, col] = 255
    light_img = img.astype(np.uint8)
    return light_img


def histogram_equalization(image):
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(image)

    return equalized_image


def apply_histogram_equalization_to_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Loop through each file in the input folder
    for file_name in files:
        # Get the full path of the file
        input_file_path = os.path.join(input_folder, file_name)
        if not input_file_path.endswith(('png', 'jpg', 'jpeg')):
            continue
        # Read the image
        image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
        # Apply histogram equalization
        image = image.astype(np.float32) / 255.0

        # Ensure that the mean of the image is 0.5
        mean_pixel_value = np.mean(image)
        image -= (mean_pixel_value - 0.5)

        # Convert image back to uint8
        image = (image * 255).clip(0, 255).astype(np.uint8)

        # Apply histogram equalization
        equalized_image = histogram_equalization(image)

        # Generate the output file path
        output_file_path = os.path.join(output_folder, file_name)

        # Write the equalized image to the output folder
        cv2.imwrite(output_file_path, equalized_image)


def apply_ofir_lighting_to_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Loop through each file in the input folder
    for file_name in files:
        # Get the full path of the file
        input_file_path = os.path.join(input_folder, file_name)
        if not input_file_path.endswith(('png', 'jpg', 'jpeg')):
            continue
        # Read the image
        image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
        # Apply histogram equalization
        image = image.astype(np.float32) / 255.0
        avg_gray = np.mean(image)
        mult = 0.5 / avg_gray
        grayscale_image = lightened_img(image, mult)  # balances the light in the frame
        grayscale_image = cv2.equalizeHist(grayscale_image)  # equalizing histogram to avoid extreme light
        blur = 0.01 * cv2.GaussianBlur(grayscale_image, (5, 5), 0)
        blur = blur.astype(np.uint8)
        grayscale_image -= blur  # sharpen the img
        # Generate the output file path
        output_file_path = os.path.join(output_folder, file_name)

        # Write the equalized image to the output folder
        cv2.imwrite(output_file_path, grayscale_image)


# Path to the input folder containing images
for emotion in ['happy', 'angry', 'surprised', 'sad']:
    input_folder_path = f'/Users/omerhazan/Desktop/personal/studies/image processing/image_processing_project/archive/test/{emotion}_sorted'

    # Path to the output folder where equalized images will be saved
    output_folder_path = input_folder_path + '_ofir_light'

    # Apply histogram equalization to images in the input folder
    apply_ofir_lighting_to_folder(input_folder_path, output_folder_path)
