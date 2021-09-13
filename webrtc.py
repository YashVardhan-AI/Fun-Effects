import cv2
from helper.compied import funcmain, draw_all
from helper.face_detector import get_face_detector, find_faces, draw_faces
from helper.face_landmarks import get_landmark_model
import streamlit as st
from helper.info import about, welcome
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
import gc

st.set_page_config(
    page_title='Face Features and Landmarks Detection')

st.title('Facial Landmarks Detection App')
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Select page:", options=[
                            "Welcome", "Face Detection", "Edge Detection", "About"])


face_model = get_face_detector()
landmark_model = get_landmark_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold1 = 100
        self.threshold2 = 200

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.cvtColor(
            cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR
        )

        return img

class VideoTransformer2(VideoTransformerBase):
    def __init__(self):
        self.threshold = 120
        

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            rects = find_faces(img, face_model)
            
            for rect in rects:
                img = draw_faces(img, rects)
                cxl, cyl, cxr, cyr, points, points2, points3, points4, thresh = funcmain(img, landmark_model, rect, self.threshold)
                img = draw_all(img, cxl, cyl, cxr, cyr, points,points2, points3, points4)
        except Exception as e:
            print(e)

        return img



if page == "Face Detection":
    ctxt = webrtc_streamer(client_settings = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),key="example2", video_transformer_factory=VideoTransformer2)  
    if ctxt.video_transformer:
        ctxt.video_transformer.threshold = st.slider("Eye tracking Threshold", 0, 255, 120)




if page == "Edge Detection":
    ctx = webrtc_streamer(client_settings = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),key="example", video_transformer_factory=VideoTransformer)
    if ctx.video_transformer:
        ctx.video_transformer.threshold1 = st.slider("Threshold1", 0, 1000, 100)
        ctx.video_transformer.threshold2 = st.slider("Threshold2", 0, 1000, 200)
    
collected = gc.collect()
print("Garbage collector: collected",
          "%d objects." % collected)

if page == 'Welcome':
    st.header("Welcome to the Neural Network based facial landmarks detection App!")
    welcome()


if page == 'About':
    st.header("About section")
    about()