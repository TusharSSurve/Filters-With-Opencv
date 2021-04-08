import cv2 
import numpy as np


def clean(img,imgSnow,pts):
    img_mask = np.zeros_like(img)
    cv2.circle(img_mask,(pts[0],pts[1]),pts[2],(255,255,255),-1)
    result = cv2.bitwise_and(img,img_mask)

    img_snow_mask = np.ones_like(imgSnow)
    img_snow_mask.fill(255)

    cv2.circle(img_snow_mask,(pts[0],pts[1]),pts[2],(0,0,0),-1)
    result_snow = cv2.bitwise_and(imgSnow,img_snow_mask)
    return cv2.bitwise_or(result,result_snow)
