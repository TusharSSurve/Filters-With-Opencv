import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
nose_image = cv2.imread("Resources/pig_nose.png")
success, img = cap.read()
rows, cols, _ = img.shape
nose_mask = np.zeros((rows, cols), np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Filter - Pig Nose/shape_predictor_68_face_landmarks.dat")

while True:
    success, img = cap.read()
    nose_mask.fill(0)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img)
    for face in faces:
        landmarks = predictor(imgGray, face)

        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0],left_nose[1] - right_nose[1]) * 1.7)
        nose_height = int(nose_width * 0.77)

        top_left = (int(center_nose[0] - nose_width / 2),int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),int(center_nose[1] + nose_height / 2))

        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = img[top_left[1]: top_left[1] + nose_height,top_left[0]: top_left[0] + nose_width]
        nose_no_area = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)

        img[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = cv2.add(nose_no_area, nose_pig)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break