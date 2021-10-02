import cv2 as cv
import numpy as np


effect_names=['cartoonify', 'watercolor', 'canny', 'pencil', 'econify', 'negative', 'faces', 'surprise']

def canny_img(img):
    """
    Canny edge detection
    """
    img = cv.Canny(img, 75, 120)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def cartoonify(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)
    edges = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 10)
  # Making a Cartoon of the image
    color = cv.bilateralFilter(frame, 12, 250, 250)
    cartoon = cv.bitwise_and(color, color, mask=edges)
    cartoon_image = cv.stylization(frame, sigma_s=150, sigma_r=0.25)
    frame = cartoon_image
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame


def watercolor(frame):
    frame = cv.stylization(frame, sigma_s=60, sigma_r=0.6)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame


def pencil(frame):
    pencil, color = cv.pencilSketch(frame, sigma_s=60, sigma_r=0.5, shade_factor=0.010)
    frame = cv.cvtColor(pencil, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480)) 
    return frame

def econify(frame):
    canny = canny_img(frame)

    blue, g, r = cv.split(canny) 
    blank = np.zeros(canny.shape[:2], dtype='uint8')

    green = cv.merge([blank,g,blank])
        
    frame = green
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame

def negative(frame):
    frame = cv.bitwise_not(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (640, 480))
    return frame





    
