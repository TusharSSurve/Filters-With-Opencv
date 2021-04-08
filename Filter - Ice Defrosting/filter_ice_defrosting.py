import cv2 
import mediapipe as mp 
import numpy as np 
from effect import clean
 

imgSnow = cv2.imread('Resources/snow.jpg')
imgSnow = cv2.resize(imgSnow,(640,480))
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

mpHands = mp.solutions.hands 
hands = mpHands.Hands(min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils
count = 0
points = []

while True:
    success,img = cap.read()
    img1 = img.copy()
    imgRGB = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    finalImg = cv2.addWeighted(img1,1,imgSnow,0.6,0)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for lm in handlandmark.landmark:
                h,w,_ = img1.shape
                lmList.append([int(lm.x*w),int(lm.y*h)])
            x,y,w,h = cv2.boundingRect(np.array(lmList))
            points.append([x + w // 2, y + h // 2,h // 2])
        count = 0
    else:
        if count<30:
            count+=1
        else:
            finalImg = cv2.addWeighted(img,1,imgSnow,0.6,0)
            points = []
            count=0

    if len(points)!=0:
        for pts in points:
            finalImg = clean(img,finalImg,pts)
    cv2.imshow('Image',finalImg)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

