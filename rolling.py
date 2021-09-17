import cv2 as cv
import numpy as np
import streamlit as st
import time


def roll():
    if st.button('start'):
        stop = st.button('stop')
        frame = st.empty()
        cap = cv.VideoCapture('Face_detection/roll.mp4')
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (640, 480))
            else:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            

            frame.image(image)
            time.sleep(0.008)
            if stop:
                break
        cap.release()
        
