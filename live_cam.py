from faceAnalizer import FaceAnalizer
from handAnalizer import HandAnalizer
from emotionAnalizer import EmotionAnalizer

import cv2
import sys
import time

MODEL_PATH = "./transfer_learning_model.h5"

faceAnalizer = FaceAnalizer()
handAnalizer = HandAnalizer()
emotionAnalizer = EmotionAnalizer(MODEL_PATH)

def live_detection(video_link = 0):

    cap = cv2.VideoCapture(video_link)

    while True:
        ret, frame = cap.read()

        # frame = faceAnalizer.live_detection(frame, part= 'all')
        frame = emotionAnalizer.live_detection(frame)
        frame = handAnalizer.live_detection(frame)
        

    
        cv2.imshow("Live Cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(1)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        live_detection()
    else:
        video_link = sys.argv[1]
        live_detection(video_link)
    


