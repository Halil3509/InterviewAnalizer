import cv2 
import mediapipe as mp
import math
from Analizer import AbstractAnalizer
import utils
import numpy as np

LANDMARKS_PATH = 'landmark_coords_hand.yaml'

class HandDetector(AbstractAnalizer):
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.landmarks_coords = utils.get_yaml(LANDMARKS_PATH)
        self.fingers = []
        self.lmList = []


    def findHands(self, img):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        allHands = []
        if img is not None or len(img) != 0:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            
            h, w, c = img.shape
            if self.results.multi_hand_landmarks:
                for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                    myHand = {}
                    ## lmList
                    mylmList = []
                    xList = []
                    yList = []
                    for id, lm in enumerate(handLms.landmark):
                        px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                        mylmList.append([px, py, pz])
                        xList.append(px)
                        yList.append(py)

                    ## bbox
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    boxW, boxH = xmax - xmin, ymax - ymin
                    bbox = xmin, ymin, boxW, boxH
                    cx, cy = bbox[0] + (bbox[2] // 2), \
                            bbox[1] + (bbox[3] // 2)

                    myHand["lmList"] = mylmList
                    myHand["bbox"] = bbox
                    myHand["center"] = (cx, cy)

                    allHands.append(myHand)

        return allHands
    

    def find_single_distance(self, p1, p2):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                    Image with output drawn
                    Line information
        """
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        
        return length
    

    def find_distance(self, hands):
        
        dist_total = 0

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            distance_dict = dict()
            distance_dict['dist1'] = []
            distance_dict['dist2'] = []
            distance_dict['dist3'] = []
            distance_dict['dist4'] = []

            # get mean for one hand
            for i in range(len(self.landmarks_coords["hand"]) - 1):
                dist_total += self.find_single_distance(lmList1[self.landmarks_coords["hand"][i]][0:2], lmList1[self.landmarks_coords["hand"][i + 1]][0:2])

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points

                # get mean
                for i in range(len(self.landmarks_coords["hand"]) - 1):
                    dist_total += self.find_single_distance(lmList2[self.landmarks_coords["hand"][i]][0:2], lmList2[self.landmarks_coords["hand"][i + 1]][0:2])

                dist_mean = np.round((dist_total / 8), 2)

            else:
                dist_mean = np.round((dist_total / 4), 2)

            return dist_mean
        
        return False
                    
                
    # override method
    def draw(self, values, img):
        """
        Draw image and return drawn image

        @param values: hands 
        """
        if len(values) == 2 or len(values) == 1:
            for hand in values:
                landmarks = hand['lmList']

                for index in self.landmarks_coords["hand"]:
                    landmark = landmarks[index]  # Extract x and y coordinates
                    x, y = landmark[0], landmark[1] 
                    cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
        
        return img 


    def calc_change(self, new_dist, old_dist):
        """
        abs(new_dist - old_dist) / old_dist
        """
        if old_dist == 0:
            result = 0
        else:
            result = np.abs(new_dist - old_dist) / old_dist

        return result











