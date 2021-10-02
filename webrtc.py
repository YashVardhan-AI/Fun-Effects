import streamlit as st
from helper.info import about, welcome
from streamlit_webrtc import webrtc_streamer, ClientSettings
from helper.effects import effect_names
from rolling import roll
from helper.video_transformer import get_video_transformer
from helper.face_detector import get_face_detector


face_model = get_face_detector()

st.set_page_config(
    page_title="Face Features and Landmarks Detection"
)

st.title("Facial Landmarks Detection App")
st.sidebar.title('Navigation')

page = st.sidebar.selectbox(
    "Select page:", 
    options = ["Welcome", "Effects", "About"]
)


if page == "Effects":
    effect_name = st.sidebar.selectbox("Choose the style model: ", effect_names)
    
    if effect_name == "surprise":
        roll()
    else:
        VideoTransformer = get_video_transformer(effect_name, face_model)

        ctxt = webrtc_streamer(
            client_settings = ClientSettings(
                rtc_configuration = {
                    "iceServers": [{
                        "urls": ["stun:stun.l.google.com:19302"]
                    }]
                },
                media_stream_constraints = {
                    "video": True, 
                    "audio": False
                },
            ),
            key = "effects", 
            video_transformer_factory = VideoTransformer
        )  

if page == 'Welcome':
    st.header("Welcome to the Neural Network based facial landmarks detection App!")
    welcome()

if page == 'About':
    st.header("About section")
    about()
