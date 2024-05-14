from pathlib import Path
import os
import cv2
import time
import depthai as dai
from base_camera import BaseCamera
import numpy as np
import argparse


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        nnPathDefault = str((Path(__file__).parent / Path('./models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
        parser = argparse.ArgumentParser()
        parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
        parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

        args = parser.parse_args()

        fullFrameTracking = args.full_frame

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        objectTracker = pipeline.create(dai.node.ObjectTracker)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        trackerOut = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        spatialDetectionNetwork.setBlobPath(args.nnPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        camRgb.preview.link(spatialDetectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
        objectTracker.out.link(trackerOut.input)

        if fullFrameTracking:
            camRgb.setPreviewKeepAspectRatio(False)
            camRgb.video.link(objectTracker.inputTrackerFrame)
            objectTracker.inputTrackerFrame.setBlocking(False)
            # do not block the pipeline if it's too slow on full frame
            objectTracker.inputTrackerFrame.setQueueSize(2)
        else:
            spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            while(True):
                imgFrame = preview.get()
                track = tracklets.get()

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                frame = imgFrame.getCvFrame()
                trackletsData = track.tracklets
                for t in trackletsData:
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    try:
                        label = labelMap[t.label]
                    except:
                        label = t.label

                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                # cv2.imshow("tracker", frame)

                # if cv2.waitKey(1) == ord('q'):
                #     break

                yield cv2.imencode('.jpg', frame)[1].tobytes()
                # cv2.imshow("tracker", trackerFrame)

                # if cv2.waitKey(1) == ord('q'):
                #     break

        # with dai.Device() as device:
        #     # Device name
        #     print('Device name:', device.getDeviceName())
        #     # Bootloader version
        #     if device.getBootloaderVersion() is not None:
        #         print('Bootloader version:', device.getBootloaderVersion())
        #     # Print out usb speed
        #     print('Usb speed:', device.getUsbSpeed().name)
        #     # Connected cameras
        #     print('Connected cameras:', device.getConnectedCameraFeatures())

        #     # Create pipeline
        #     pipeline = dai.Pipeline()
        #     cams = device.getConnectedCameraFeatures()
        #     streams = []
        #     for cam in cams:
        #         print(str(cam), str(cam.socket), cam.socket)
        #         c = pipeline.create(dai.node.Camera)
        #         x = pipeline.create(dai.node.XLinkOut)
        #         c.preview.link(x.input)
        #         c.setBoardSocket(cam.socket)
        #         stream = str(cam.socket)
        #         if cam.name:
        #             stream = f'{cam.name} ({stream})'
        #         x.setStreamName(stream)
        #         streams.append(stream)

        #     # Start pipeline
        #     device.startPipeline(pipeline)
        #     fpsCounter = {}
        #     lastFpsCount = {}
        #     tfps = time.time()
        #     while not device.isClosed():
        #         queueNames = device.getQueueEvents(streams)
        #         for stream in queueNames:
        #             messages = device.getOutputQueue(stream).tryGetAll()
        #             fpsCounter[stream] = fpsCounter.get(stream, 0.0) + len(messages)
        #             for message in messages:
        #                 # Display arrived frames

        #                 if type(message) == dai.ImgFrame:
        #                     # render fps
        #                     print(messages.index(message))
        #                     fps = lastFpsCount.get(stream, 0)
        #                     frame = message.getCvFrame()
                            
        #                     cv2.putText(frame, "Fps: {:.2f}".format(fps), (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
        #                     # cv2.imshow(stream, frame)

        #                     yield cv2.imencode('.jpg', frame)[1].tobytes()       
        #         ## Pseudo for future reference     
        #         # stream = queueNames[0]
        #         # messages = device.getOutputQueue(stream).tryGetAll()
        #         # if(messages[0] == dai.ImgFrame):
        #         #     # render fps
        #         #     fps = lastFpsCount.get(stream, 0)
        #         #     frame = messages[0].getCvFrame()
                    
        #         #     cv2.putText(frame, "Fps: {:.2f}".format(fps), (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
        #         #     frame = messages[0].getCvFrame()
        #         #     yield cv2.imencode('.jpg', frame)[1].tobytes()   
        #         if time.time() - tfps >= 1.0:
        #             scale = time.time() - tfps
        #             for stream in fpsCounter.keys():
        #                 lastFpsCount[stream] = fpsCounter[stream] / scale
        #             fpsCounter = {}
        #             tfps = time.time()

        #         # if cv2.waitKey(1) == ord('q'):
        #         #     break


        # # camera = cv2.VideoCapture(Camera.video_source)
        # # if not camera.isOpened():
        # #     raise RuntimeError('Could not start camera.')

        # # while True:
        # #     # read current frame
        # #     _, img = camera.read()

        # #     # encode as a jpeg image and return it
        # #     yield cv2.imencode('.jpg', img)[1].tobytes()

 