import streamlit as st


def about():
    st.markdown(
        """
        ## How does it work? 
        The Method Used for Detection of Facial Features uses [TensorFlow](https://www.tensorflow.org/) and [OpenCV](https://www.opencv.org). Here TensorFlow model is used for detecting a 72 point face mash from which the points that correspond to the facial features can be used to detect those landmarks. OpenCV is used to Detect and do Eye Tacking.
        
        ### Below are the guides and Tutorials that helped do the project:
        - The original repository for the model on github [yinguobing/cnn-facial-landmarks](http://github.com/yinguobing/cnn-facial-landmarks).
        - [Eye tracking guide](https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6).
        - The [Comparision of different face detection techniques](https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c).

        ## This Project on GitHub [YashVardhan-AI/Fun-Effects](https://github.com/YashVardhan-AI/Fun-Effects)
        """
    )

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
 