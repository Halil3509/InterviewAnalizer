import cv2 as cv
import mediapipe as mp
import time

import numpy as np
import utils
import validation as val_file
from tqdm import tqdm
from Analizer import AbstractAnalizer

# constants 
FONTS =cv.FONT_HERSHEY_COMPLEX
LANDMARKS_PATH = 'landmark_coords_fasexp.yaml'

class FaceAnalizer(AbstractAnalizer):

    def __init__(self):
        self.map_face_mesh = mp.solutions.face_mesh
        self.landmarks_coords = utils.get_yaml(LANDMARKS_PATH)


    @staticmethod
    def fillPolyTrans(img, points, color = (255,255,0), opacity = 0.4):
        """
        @param img: (mat) input image, where shape is drawn.
        @param points: list [tuples(int, int) these are the points custom shape,FillPoly
        @param color: (tuples (int, int, int)
        @param opacity:  it is transparency of image.
        @return: img(mat) image with rectangle draw.

        """
        list_to_np_array = np.array(points, dtype=np.int32)
        overlay = img.copy()  # coping the image
        cv.fillPoly(overlay,[list_to_np_array], color )
        new_img = cv.addWeighted(overlay, opacity, img, 1 - opacity, 0)

        img = new_img
        cv.polylines(img, [list_to_np_array], True, color,1, cv.LINE_AA)
        return img


    @staticmethod
    def find_landmark(img, results):
        """
        Detects landmark
        """

        img_height, img_width= img.shape[:2]
         # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        
        return mesh_coord
    
    
    def draw(self, part_values, frame):
        """
        Finds specified landmarks and draws 

        @param part_values: wanted part values
        @param frame: image
        @return: situation (everthings ok?), drawn image, landmarks coordinates 
        """
        
        with self.map_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
            
            try:
                results = face_mesh.process(frame)
                
            except AttributeError:
                return False,"",""
            
            drawn_image = frame.copy()
            mesh_coords = []
            if results.multi_face_landmarks:
                    mesh_coords = self.find_landmark(frame, results)

                    drawn_image = self.fillPolyTrans(frame, [mesh_coords[p] for p in part_values])

                    [cv.circle(drawn_image, mesh_coords[p], 1,(255,255,0),  -1, cv.LINE_AA) for p in part_values]

            if len(mesh_coords) == 0:
                return False, "", ""

        return True, drawn_image, [mesh_coords[p] for p in part_values]
    
    
    def detect(self, frame,  part_name= 'face'):
        """
        Detects specified part of head

        @param frame = image

        """

        # validation part_name parameter
        all_situation = val_file.part_name_control(part_name, LANDMARKS_PATH)
        
        
        if all_situation:
            concatenated_array = []
            for value in self.landmarks_coords.values():
                concatenated_array.extend(value)

            sit, drawn_image, results = self.draw(concatenated_array, frame)
            
        else:
            sit, drawn_image, results = self.draw(self.landmarks_coords[part_name], frame)


        return sit, drawn_image, results

        

    def gesfacexp_score(self, reference_image, video_path, fps = 5):
        """
        Calculates gesfacexp_score

        @param reference_image = first taken image
        @param video_path = selected video's path
        @param fps = frame per second
        
        @return dict format gesfacexp_score = [eye_gesfac_score, lips_gesfac_score, eyebrow_gesfac_score, total_gesfac_score]
        """

        ## Reference Image is readable?
        if not self.readable(reference_image):
            raise AttributeError("Reference Image is not readable")
        

        cap = cv.VideoCapture(video_path)

        number_scores = dict()
        number_scores['eye'] = 0
        number_scores['lips'] = 0
        number_scores['eyebrow'] = 0

        total_gesfac_scores = dict()
        total_gesfac_scores['eye']= []
        total_gesfac_scores['lips'] = []
        total_gesfac_scores['eyebrow'] = []

        total_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total = total_frame, unit = 'frame')

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # Readable Control  
            if self.readable(frame):
                eye_results_mean, lips_results_mean, eyebrow_results_mean = self.get_mean_multiple(frame)

                if np.shape(eye_results_mean)[0] != 0:
                    total_gesfac_scores['eye'].append(eye_results_mean)
                    number_scores['eye'] +=1
                
                if np.shape(lips_results_mean)[0] != 0:
                    total_gesfac_scores['lips'].append(lips_results_mean)
                    number_scores['lips'] +=1
                
                if np.shape(eye_results_mean)[0] != 0:
                    total_gesfac_scores['eyebrow'].append(eyebrow_results_mean)
                    number_scores['eyebrow'] +=1

                progress_bar.update(1)
                
                time.sleep(1/fps)
        
       
        cap.release()
        progress_bar.close()

        # Calc gesfacexp score
        gesfacexp_score = self.calcs_gesfacexp(total_gesfac_scores, number_scores, reference_image)
    
        return gesfacexp_score
    
    
    def calcs_gesfacexp(self, total_gesfac_scores:dict, number_scores: dict, reference_image):
        mean_gesfac_scores= dict()
        mean_gesfac_scores['eye'] = []
        mean_gesfac_scores['lips'] = []
        mean_gesfac_scores['eyebrow'] = []
        
        # Get Mean
        mean_gesfac_scores['eye']  = np.divide(np.sum(total_gesfac_scores['eye'], axis = 0), number_scores['eye'])
        mean_gesfac_scores['lips']  = np.divide(np.sum(total_gesfac_scores['lips'], axis = 0), number_scores['lips'])
        mean_gesfac_scores['eyebrow']  = np.divide(np.sum(total_gesfac_scores['eyebrow'], axis = 0), number_scores['eyebrow'])
        

        
        # For reference Image
        eye_results_mean, lips_results_mean, eyebrow_results_mean = self.get_mean_multiple(reference_image)

        # mean_result_image - reference_image_results and get absolute to find alteration
        eye_diff = np.abs(mean_gesfac_scores['eye'] - eye_results_mean)
        lips_diff = np.abs(mean_gesfac_scores['lips'] - lips_results_mean)
        eyebrow_diff = np.abs(mean_gesfac_scores['eyebrow'] - eyebrow_results_mean)

        # get probability for each part of body
        eye_gesfacexp_score = (np.divide(eye_diff, eye_results_mean))*100
        lips_gesfacexp_score = (np.divide(lips_diff, lips_results_mean))*100
        eyebrow_gesfacexp_score = (np.divide(eyebrow_diff, eyebrow_results_mean))*100

        # Create dict and assign scores
        gesfacexp_dict = dict()
        gesfacexp_dict["eye"] = np.mean(eye_gesfacexp_score)
        gesfacexp_dict["lips"] = np.mean(lips_gesfacexp_score)
        gesfacexp_dict["eyebrow"] = np.mean(eyebrow_gesfacexp_score)

        # Calculate gesfacexp score 
        gesfacexp_score = np.round(np.mean([gesfacexp_dict["eye"], gesfacexp_dict["lips"], gesfacexp_dict["eyebrow"]]), 3)
        
        
        
        gesfacexp_dict["total"] = gesfacexp_score

        return gesfacexp_dict


    def get_mean_multiple(self, image):
        """
        returns mean of coordinates
        """
        _, __, eye_results = self.detect(frame=image, part_name='eye')
        _, __, lips_results = self.detect(frame=image, part_name='lips')
        _, __, eyebrow_results = self.detect(frame=image, part_name='eyebrow')


        eye_results_mean = self.get_mean_single(eye_results)
        lips_results_mean = self.get_mean_single(lips_results)
        eyebrow_results_mean = self.get_mean_single(eyebrow_results)

        return eye_results_mean, lips_results_mean, eyebrow_results_mean

    @staticmethod
    def get_mean_single(results):
        """
        gets average of x, y  in one coordinate and convert to 1 dimension from 2 dimension. 
        z is not selected due to the fact that image has 2 two dimension

        @param results = detected mesh coords

        return = a list that has mean of x and y landmark coordinates 
        """
        mean_list = [(x + y)/ 2 for x, y in results]

        return mean_list
    

    def readable(self, image):
        """
        Checks that are there pieces of head

        return =  True or False
        """
        sit, _, __ = self.detect(image, part_name= 'all')

        return sit
        
 

    def live_detection(self, frame, part = "all"):
        """
        uses for live detection time
        """

        sit, image, results = self.detect(frame, part)

        return image

        
        



            

    
    