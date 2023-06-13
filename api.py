from handAnalizer import HandAnalizer
from faceAnalizer import FaceAnalizer
from emotionAnalizer import EmotionAnalizer


from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from keras.utils import load_img
import numpy as np
from keras.models import  load_model
import cv2

MODEL_PATH = "./transfer_learning_model.h5"

# Load Image Data
model = load_model(MODEL_PATH, compile=False)
optims = [tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999)]

model.compile(loss = 'categorical_crossentropy',
              optimizer = optims[0],
              metrics = ['accuracy'])


face_haar_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# Create Flask app
app = Flask(__name__)
CORS(app)


@app.route('/gesfacexp', methods=['POST'])
def gesfacexp_score():
 
    # call the class
    face_analizer = FaceAnalizer()

    # Get the image from the request
    video_link = request.get_json()['video_link']
    reference_image = request.get_json()['reference_image']
    fps = request.get_json()['fps']
    
    score =  face_analizer.gesfacexp_score(reference_image=reference_image, video_path=video_link, fps = fps)

    return jsonify({'result': score})


@app.route("/handScore", methods=['POST'])
def hand_score():

    handAnalizer = HandAnalizer()

    video_path = request.get_json()['video_path']
    fps = request.get_json()['fps']

    score = handAnalizer.get_hand_score(video_path=video_path, fps= fps)

    return jsonify({'result': score})


@app.route('/emotionScore', methods = ['POST'])
def emotion_score():
    emotionAnalizer = EmotionAnalizer(MODEL_PATH)

    video_path = request.get_json()['video_path']
    fps = int(request.get_json()['fps'])

    emotion_score = emotionAnalizer.get_emotion_score(video_path= video_path, fps = fps)

    return jsonify({'result':emotion_score})

# Run Flask app
if __name__ == '__main__':
    app.run(port=5000)