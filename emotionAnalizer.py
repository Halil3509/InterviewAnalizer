import tensorflow as tf
import cv2
import numpy as np
import time
from collections import Counter
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array

class EmotionAnalizer():

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Disgust']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    
    def load_model(self):

        model = tf.keras.models.load_model(self.model_path)

        return model 
    
    def live_detection(self, frame):
        """
        Predict emotion
        """

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region of interest
            face_roi = frame[y:y+h, x:x+w]

            # Resize the face region to 48x48 pixels
            resized = cv2.resize(face_roi, (48, 48))

            # Convert the resized face to an array
            face = img_to_array(resized)


            # Expand dimensions to match the input shape of the model
            face = np.expand_dims(face, axis=0)

            # Predict the emotion
            predictions = self.model.predict(face)

            # Convert str 
            emotion = self.emotion_labels[np.argmax(predictions)]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the emotion text
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame



    def get_emotion_score(self, video_path, fps = 5):
        """
        Detect every emotion for fps value. 
        @param video_path = video_path 
        @param fps = fps

        return get_emotion_score
        """

        # Open video file
        video = cv2.VideoCapture(video_path)

        # Parameters
        target_size = (48, 48)

        short_total_predictions = []
        emotion_number_dict = dict()
        for emotion in self.emotion_labels:
            emotion_number_dict[emotion] = 0

        # Iterate over frames
        while video.isOpened():
            # Read frame
            ret, frame = video.read()
            if not ret:
                break
            
            for i in range(fps):
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the grayscale frame
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract the face region of interest
                    face_roi = frame[y:y+h, x:x+w]

                    # Resize the face region to 48x48 pixels
                    resized = cv2.resize(face_roi, (48, 48))

                    # Convert the resized face to an array
                    face = img_to_array(resized)


                    # Expand dimensions to match the input shape of the model
                    face = np.expand_dims(face, axis=0)

                    # Predict the emotion
                    predictions = self.model.predict(face)

                    if len(short_total_predictions) == 0:
                        short_total_predictions = predictions
                    else:
                        short_total_predictions = [x + y for x, y in zip(short_total_predictions, predictions)]

                time.sleep(1/fps)


        

            


            emotion = self.emotion_labels[np.argmax(short_total_predictions)]
            # increment detected emotion 
            emotion_number_dict[emotion] +=1


        # Release video capture and close any open windows
        video.release()
    

        return emotion_number_dict

        
        

        