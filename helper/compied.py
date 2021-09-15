from helper.utils import *
import cv2
import numpy as np
from helper.face_landmarks import detect_marks
from helper.face_landmarks import draw_marks, line, linemain



def funcmain(img, rect, value, left=[36, 37, 38, 39, 40, 41], right=[42, 43, 44, 45, 46, 47], kernel = np.ones((9, 9), np.uint8)):
        try:
                shape = detect_marks(img, rect)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask = eye_on_mask(mask, left, shape)
                mask = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
        
                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
       
                _, thresh = cv2.threshold(eyes_gray, value, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)
        
                cxl, cyl=contouring(thresh[:, 0:mid], mid, img)
                cxr, cyr = contouring(thresh[:, mid:], mid, img, True)

                points = shape[27:36]
                points2 = shape[27:31]
                points3 = shape[30:36]
                points4 = shape[48:]
              
                return cxl, cyl, cxr, cyr, points, points2, points3, points4, thresh
        except:
                pass

def draw_all(img, cxl, cyl, cxr, cyr, points, points2, points3, points4):
        
        img = draw_marks(img, points4)
        img = line(img,points4)
        img = draw_marks(img, points, color=(150,100,50))
        img = draw_marks(img, points3, color=(150,100,50))
        img = line(img, points3)
        img = linemain(img, points2)
        img = cv2.circle(img, (cxl, cyl), 8, (0, 0, 255), 2)
        img = cv2.circle(img, (cxr, cyr), 8, (0, 0, 255), 2)
 
        return img