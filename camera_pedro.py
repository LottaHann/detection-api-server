import cv2 as cv
import threading
# from imutils.video.pivideostream import PiVideoStream  #Only if using a pi camera. You also need to change some code in that case.
from base_camera import BaseCamera
import time
from datetime import datetime
import numpy as np
from base64 import b64encode
from enum import Enum
import uuid
import os


class Camera(BaseCamera):
    def __init__(self, flip = False, file_type  = ".jpg", photo_string= "stream_photo", camera_index=0):
        # self.vs = PiVideoStream(resolution=(1920, 1080), framerate=30).start()
        if os.environ.get('MODULE'):
            print("Checking cameras")
            module=int(os.environ['MODULE'])
            camera_index=module

        self.camera_index = camera_index
        self.flip = flip # Flip frame vertically
        self.file_type = file_type # image type i.e. .jpg
        self.model = cv.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                      'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        self.classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


        time.sleep(2.0)
        
        # if not self.vs.isOpened():
        #     raise ValueError("Unable to open USB camera")
        super(Camera, self).__init__()

    def id_class_name(self, class_id):  # Only takes class_id
        for key, value in self.classNames.items():
            if class_id == key:
                return value



    # def __del__(self):
    #     self.vs.release()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame
    @staticmethod
    def frames():
        """Retrieves a frame from the video source, processes it, and returns it as a JPEG-encoded byte string along with person detection status. """
        camera = Camera()
        camera.vs = cv.VideoCapture(camera.camera_index)
        
        # ret, frame = self.flip_if_needed(self.vs.read())
        while True:
            ret, frame = camera.vs.read()
            if not ret:  # Check if frame was read successfully
                print("Error reading frame, check camera connection")
                return None # Or raise an exception if needed
            image_height, image_width, _ = frame.shape

            cp_frame, person_detected, tracklets = camera.detect_and_draw_person(frame)
            
            ret, jpeg = cv.imencode('.jpg', cp_frame)
            if not ret:
                print("Error encoding frame as JPEG")
                return None 
            frame_data = b64encode(jpeg).decode('utf-8')
            
            previous_frame = jpeg
            objects = []
            yield cv.imencode('.jpg', cp_frame)[1].tobytes(), tracklets, person_detected


    def release(self):
        self.vs.release()


    def detect_and_draw_person(self,frame, confidence_threshold=0.7):
        """Detects and draws bounding boxes around persons in a given frame.

        Args:
            frame: The OpenCV image frame to process.
            model: The pre-trained object detection model.
            id_class_name: A function to map class IDs to class names.
            confidence_threshold: The minimum confidence level for drawing bounding boxes.

        Returns:
            The OpenCV image frame with bounding boxes drawn around detected persons.
        """
        person_detected = False
        frame_copy = frame.copy()  # Avoid modifying the original frame
        tracklets = []
        frame_width = frame_copy.shape[1]
        frame_height = frame_copy.shape[0]
        frame_center_x = frame_width // 2

        

        blob = cv.dnn.blobFromImage(frame_copy, size=(150, 150), swapRB=True)
        self.model.setInput(blob)
        output = self.model.forward()

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > confidence_threshold:
                class_id = detection[1]
                class_name = self.id_class_name(class_id)
                if class_name == 'person':
                    person_detected = True


                                                # Calculate bounding box coordinates
                    x1 = int(detection[3] * frame_copy.shape[1])
                    y1 = int(detection[4] * frame_copy.shape[0])
                    x2 = int(detection[5] * frame_copy.shape[1])
                    y2 = int(detection[6] * frame_copy.shape[0])

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    normalized_x = (center_x - frame_center_x) / frame_center_x
                
                    # Create tracklet data
                    tracklet = {
                        "id": str(len(tracklets)),  # Generate a unique ID
                        "label": class_name,
                        "status": "Tracked",  # Assume new detection
                        "roi": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "spatialCoordinates": {
                            "x": normalized_x*1000,  # Estimate X as center of bounding box
                            "y": int((y1 + y2) // 2),  # Estimate Y as center of bounding box
                            "z": 0  # We don't have depth information, so set to 0
                        }
                    }
                    tracklets.append(tracklet)
                    


                    # print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                    # box_x = detection[3] * frame_copy.shape[1]  # image_width
                    # box_y = detection[4] * frame_copy.shape[0]  # image_height
                    # box_width = detection[5] * frame_copy.shape[1]  
                    # box_height = detection[6] * frame_copy.shape[0] 
                    # cv.rectangle(frame_copy, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                    # cv.putText(frame_copy, class_name, (int(box_x), int(box_y + 0.05 * frame_copy.shape[0])), cv.FONT_HERSHEY_SIMPLEX, (0.005 * frame_copy.shape[1]), (0, 0, 255))
                    cv.rectangle(frame_copy, (x1, y1), (x2, y2), (23, 230, 210), thickness=1)
                    cv.circle(frame_copy, (center_x, center_y), 3, (0, 255, 0), -1)
                    cv.putText(frame_copy, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame_copy, person_detected, tracklets