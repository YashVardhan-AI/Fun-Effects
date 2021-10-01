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
    st.markdown(
        """
        by [Yash Vardhan](https://github.com/YashVardhan-AI)
        
        ## Please select the page you want to navigate to:
        - Welcome (this page)
        - Effects: Effects you want to apply to the input video
            - Cartoonify
            - Negative 
            - Econify
            - Watercolor
            - Pencil
            - Canny
            - Faces
            - Surprise (a very interesting surprise)
        - About: to read more about the neural network and the algorithm behind the hooks and see how it works is about.
        """
    )
 