import cv2
from HandDetector import HandDetector
import numpy as np
import time
from tqdm import tqdm


class HandAnalizer():

    def get_hand_score(self, video_path, fps = 5, show = False):
        cap = cv2.VideoCapture(video_path)

        old_mean = 0
        number_hand_count = 0
        average_dist_change = 0
        total_frame = 0
        hand_frame = 0

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total = total_frame, unit = 'frame')

        while cap.isOpened():
            # perform fps
            time.sleep(1/fps)

            progress_bar.update(1)

            ret, frame = cap.read()

            # Check if the video capture was successful
            if not ret:
                break

            #inc total frame 
            total_frame += 1
            
            # Call HandDetector Class 
            detector = HandDetector()

            # Find hand(s) coordinats
            hands = detector.findHands(frame)


            if len(hands) != 0:
                # inc hand frame count 
                hand_frame += 1 

                # Find mean of distance
                new_mean = detector.find_distance(hands)

                # calculate change quality of hand
                dist_change_score = detector.calc_change(new_mean, old_mean) *100

                # new will be old anymore right here
                old_mean = new_mean

                # first value
                if number_hand_count == 0:
                    average_dist_change = dist_change_score

                number_hand_count += 1

                if number_hand_count == 2:
                    # one of them is average value. Other one will be coming value
                    number_hand_count = 1
                    average_dist_change = np.mean([average_dist_change, dist_change_score]) 


        progress_bar.close()
        cap.release()  

        # Calculate hand_score 
        hand_score_dict = self._calcs_hand_score(average_dist_change, hand_frame, total_frame)

        return hand_score_dict

    def _calcs_hand_score(self, average_dist_change, hand_frame, total_frame):
        """
        Calcs hand score using that formula 

        average_dist_change * (hand_frame / total_frame)
        """
        hand_score = np.round((average_dist_change * (hand_frame / total_frame)), 2)
        
        result = dict()
        result["hand_score"] = hand_score
        result["frame_ratio"] = np.round((hand_frame / total_frame), 2)
        result["average_dist_change"] = np.round(average_dist_change, 2)

        return result
    
    def live_detection(self, image):

        detector = HandDetector()

        # Find hand(s) coordinats
        hands = detector.findHands(image)

        if hands:
            image =  detector.draw(hands, image)

        return image


    
