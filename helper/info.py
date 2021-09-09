import streamlit as st

def about():
    st.markdown("""### The Method I used for detecting The Facial features uses Tensorflow and OpenCV. I use Tensorflow model for detecting a 72 point face mash from which the points that corresspond to the facial features can be used to detect those landmarks.  I also use Opencv to Detect and do Eye tracking.""")
    st.markdown("### below are the guides and Tutorials that helped do the project")
    st.markdown('Link to the orignal github repository for the model >><a href="https://github.com/yinguobing/cnn-facial-landmark" target="_blank">The github link</a>', unsafe_allow_html=True)
    st.markdown('link to the eye tracking guide>> <a href="https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6" target="_blank">The eye tracking guide</a>', unsafe_allow_html=True)
    st.markdown('link to the comparison of different face Detection techniques>> <a href="https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c" target="_blank">The comparison</a>', unsafe_allow_html=True)
    st.markdown('### Link to my github repository>> <a href="https://github.com/porus-creator/cotm_3" target="_blank">Click Here</a> ', unsafe_allow_html=True )


def welcome():
    st.markdown("By Yash Vardhan")
    st.markdown("## Please select the page you want to navigate to.")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("""### Face Detection : try the app, you can choose your own images for both style and content, or try the pre-loaded ones.""")
    st.markdown("""- Threshold slider allows you to change the  threshhold value""")
    st.markdown("""- The show threshold checkbox allows you to see the thresholded image""")    
    st.markdown("  ")
    st.markdown("### edge detection : In this page you can detect all the edges in a image")
    st.markdown("- You Can do it in real time or upload an Image")
    st.markdown("- You Can also change the values of threshold 1 and 2 to get a better result")
    st.markdown("###        About: to read more about the neural network and the algorithm behind the hooks and see how it works is about.")
    imglist =[ 'Face_detection/img/obama.jpg',
    'Face_detection/processed/edited_obama.jpg']
    st.image(imglist[0], width=300)
    st.image(imglist[1], width = 300)
    
    