from webrtc_streamer import VideoTransformerBase
from .effects import *
from .compied import funcmain, draw_all
from .face_detector import find_faces, draw_faces

def get_video_transformer(effect_name, face_model):
    class VideoTransformer(VideoTransformerBase):
        effect_name = effect_name  # maybe redundant here?

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            if effect_name == "cartoonify": img = cartoonify(img)
            elif effect_name == "negative": img = negative(img)
            elif effect_name == "econify": img = econify(img)
            elif effect_name == "watercolor": img = watercolor(img)
            elif effect_name == "pencil": img = pencil(img)
            elif effect_name == "canny": img = canny_img(img)    
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

    return VideoTransformer
