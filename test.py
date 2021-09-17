import cv2 as cv
import numpy as np
def canny_img(img):
    """
    Canny edge detection
    """
    return cv.Canny(img, 100, 120)


def econify(frame):
    canny = canny_img(frame)
    cv.imwrite("canny.jpg", canny)
    img = cv.imread("canny.jpg")
    blue, g, r = cv.split(img) 
    blank = np.zeros(img.shape[:2], dtype='uint8')

    green = cv.merge([blank,g,blank])
        
    frame = green
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    
    return frame

img = cv.imread('Face_detection/img/obama.jpg')
img = econify(img)