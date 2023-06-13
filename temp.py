import os
from faceAnalizer import FaceAnalizer
import numpy as np
import matplotlib.pyplot as plt

face_analizer = FaceAnalizer()

import cv2
img = cv2.imread('../../ben.jpeg')

video_link = './temp_video.mp4'

score =  face_analizer.gesfacexp_score(reference_image=img, video_path=video_link, fps = 5)

print("Score", score)