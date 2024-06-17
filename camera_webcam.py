import os
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('WEBCAM_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['WEBCAM_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                      'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
    
        classNames = {0: 'background',
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


        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            ret, frame = camera.read()

            if not ret:  # Check if frame was read successfully
                print("Error reading frame, check camera connection")
                return None # Or raise an exception if needed

            person_detected = False
            frame_copy = frame.copy()  # Avoid modifying the original frame
            objects = []
            blob = cv2.dnn.blobFromImage(frame_copy, size=(150, 150), swapRB=True)
            model.setInput(blob)
            output = model.forward()
            debug_data = []

            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > 0.7:
                    class_id = detection[1]
                    class_name = ''
                    for key, value in classNames.items():
                         if class_id == key:
                             class_name = value
                            

                    
                    if class_name == 'person':
                        person_detected = True
                        # print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
                        box_x = detection[3] * frame_copy.shape[1]  # image_width
                        box_y = detection[4] * frame_copy.shape[0]  # image_height
                        box_width = detection[5] * frame_copy.shape[1]  
                        box_height = detection[6] * frame_copy.shape[0] 
                        cv2.rectangle(frame_copy, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                        cv2.putText(frame_copy, class_name, (int(box_x), int(box_y + 0.05 * frame_copy.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, (0.005 * frame_copy.shape[1]), (0, 0, 255))
                        debug_data.append("box y:"+ str(box_y) )
                        debug_data.append("box x:"+ str(box_x) )
                        debug_data.append("box_width: "+str(box_width))
                        debug_data.append("height: "+str(box_height))
                        debug_data.append("middle of image: "  +str(frame_copy.shape[1]//2))
                        objects.append({
                            "id": "0",
                            "label": "person",
                            "status": "FALSE",
                            "roi": {
                                "x1": "0",
                                "y1": "0",
                                "x2": "0",
                                "y2": "0"
                            },
                            "spatialCoordinates": {
                                "x": str(box_x),
                                "y": str(box_y),
                                "z": int("400")
                            }
                        })

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame_copy)[1].tobytes(), objects, debug_data
