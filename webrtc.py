import streamlit as st
from helper.info import about, welcome
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings
from helper.effects import *
from rolling import roll
from helper.compied import funcmain, draw_all
from helper.face_detector import get_face_detector, find_faces, draw_faces

face_model = get_face_detector()

st.set_page_config(
    page_title='Face Features and Landmarks Detection')

st.title('Facial Landmarks Detection App')
st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Select page:", options=[
                            "Welcome", "Effects", "About"])




if page == "Effects":
    effect_name = st.sidebar.selectbox("Choose the style model: ", effect_names)
    
    if effect_name == "surprise":
        roll()
    else:
        class VideoTransformer(VideoTransformerBase):
            effect_name = effect_name
        
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                if effect_name == "cartoonify":
                    img = cartoonify(img)
                elif effect_name == "negative":
                    img = negative(img)
                elif effect_name == "econify":
                    img = econify(img)
                elif effect_name == "watercolor":
                    img = watercolor(img)
                elif effect_name == "pencil":
                    img = pencil(img)
                    
                elif effect_name == "canny":
                    img = canny_img(img)
                
                
                elif effect_name == "faces":
                    try:
                        rects = find_faces(img, face_model)
                    
                        for rect in rects:
                            img = draw_faces(img, rects)
                            cxl, cyl, cxr, cyr, points, points2, points3, points4, thresh = funcmain(img, rect, 120)
                            img = draw_all(img, cxl, cyl, cxr, cyr, points,points2, points3, points4)
                            
                    except Exception as e:
                        print(e)
                

                
                return img


        ctxt = webrtc_streamer(client_settings = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},),key="effects", video_transformer_factory=VideoTransformer)  





if page == 'Welcome':
    st.header("Welcome to the Neural Network based facial landmarks detection App!")
    welcome()


if page == 'About':
    st.header("About section")
    about()

