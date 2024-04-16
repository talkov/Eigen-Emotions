import cv2 as cv
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


def find_face(img, cascade):
    detected_faces = cascade.detectMultiScale(img, minNeighbors=14)
    if len(detected_faces) == 0:
        detected_faces = cascade.detectMultiScale(img, minNeighbors=8)
        if len(detected_faces) == 0:
            detected_faces = cascade.detectMultiScale(img, minNeighbors=4)
            if len(detected_faces) == 0:
                detected_faces = cascade.detectMultiScale(img, minNeighbors=2)
                if len(detected_faces) == 0:
                    print('empty')
    return detected_faces


def face_detection(frame):
    face_cascade = cv.CascadeClassifier(
        '/Users/omerhazan/Desktop/personal/studies/image '
        'processing/image_processing_project/face_detection/haarcascade_frontalface_alt.xml')
    """ sets the viola-jones algorythm to face_cascades"""
    profile_cascade = cv.CascadeClassifier(
        '/Users/omerhazan/Desktop/personal/studies/image '
        'processing/image_processing_project/face_detection/haarcascade_profileface.xml')

    # the viola-jones algorithm
    frame = cv.resize(frame, (360, 200))  # resizing the frame for easier face-recognition with live cam
    grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rescaled_img = grayscale_image / 255  # sets the pixels between (0,1)
    avg_gray = np.mean(rescaled_img)
    mult = 0.5 / avg_gray
    grayscale_image = lightened_img(rescaled_img, mult)  # balances the light in the frame
    """now we run the viola-jones algorythm to detect faces in the frame
      using multiple options for the neighbors to avoid false positive and false negative
    """

    detected_faces = find_face(grayscale_image, face_cascade)
    """
    detected_faces = face_cascade.detectMultiScale(grayscale_image,minNeighbors=8)
    if len(detected_faces) == 0:
    detected_faces = face_cascade.detectMultiScale(grayscale_image,minNeighbors=4)
    if len(detected_faces) == 0:
      detected_faces = face_cascade.detectMultiScale(grayscale_image,minNeighbors=2)
      if len(detected_faces) == 0:
        print('empty')
    """
    """gets the rectangles of the detected faces, left upper corner and height
    and width of the rectangle. minNeighbors sets the amount of overlapping rectangles
    """
    grayscale_flip = cv.flip(grayscale_image, 1)
    if len(detected_faces) == 0:
        detected_faces = find_face(grayscale_image, profile_cascade)
        if len(detected_faces) == 0:
            detected_faces = find_face(grayscale_flip, profile_cascade)

    faces = []
    for (column, row, width, height) in detected_faces:
        # cv.rectangle(frame,(column, row),
        #   (column + width, row + height),(0, 0, 0),2 )
        face = grayscale_image[row:row + height, column:column + width]
        # cuts the rectangle of the detected face
        face = cv.resize(face, (48, 48))  # resizes the img to the right size
        face = cv.equalizeHist(face)  # equalizing histogram to avoid extreme light
        face_blur = 0.01 * cv.GaussianBlur(face, (5, 5), 0)
        face_blur = face_blur.astype(np.uint8)
        face -= face_blur  # sharpen the img
        faces.append(face)

    return faces, detected_faces
