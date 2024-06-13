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
        parser.add_argument('-d', '--debug', action="store_true", help="Get debug sysInfo from camera", default=True)

        args = parser.parse_args()

        fullFrameTracking = args.full_frame
        debug = args.debug

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        objectTracker = pipeline.create(dai.node.ObjectTracker)
        
        # Camera Info
        sysLog = pipeline.create(dai.node.SystemLogger)
        linkOut = pipeline.create(dai.node.XLinkOut)

        linkOut.setStreamName("sysinfo")

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

        sysLog.setRate(25) # 1 hz

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

        sysLog.out.link(linkOut.input)

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
            
            # Output queue will be used to get the system sysInfo

            qSysInfo = device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)


            while(True):
                imgFrame = preview.get()
                track = tracklets.get()
                output = ""
                objects = []
                counter+=1
                current_time = time.monotonic()

                sysInfo = qSysInfo.get()  # Blocking call, will wait until a new data has arrived

                # self.printSystemInformation(sysInfo)

                if(debug):


                    m = 1024 * 1024  # MiB
                    output = f"Ddr used / total - {sysInfo.ddrMemoryUsage.used / m:.2f} / {sysInfo.ddrMemoryUsage.total / m:.2f} MiB\n"
                    output += f"Cmx used / total - {sysInfo.cmxMemoryUsage.used / m:.2f} / {sysInfo.cmxMemoryUsage.total / m:.2f} MiB\n"
                    output += f"LeonCss heap used / total - {sysInfo.leonCssMemoryUsage.used / m:.2f} / {sysInfo.leonCssMemoryUsage.total / m:.2f} MiB\n"
                    output += f"LeonMss heap used / total - {sysInfo.leonMssMemoryUsage.used / m:.2f} / {sysInfo.leonMssMemoryUsage.total / m:.2f} MiB\n"
                    t = sysInfo.chipTemperature
                    output += f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa: {t.upa:.2f}, dss: {t.dss:.2f}\n"
                    output += f"Cpu usage - Leon CSS: {sysInfo.leonCssCpuUsage.average * 100:.2f}%, Leon MSS: {sysInfo.leonMssCpuUsage.average * 100:.2f} %\n"
                    output += "----------------------------------------\n"
                    
                    # print(f"\033[2J\033[1;1H{output}", end="", flush=True,)             
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
                    tracklet_data = {
                        "id": t.id,
                        "label": label,
                        "status": t.status.name,
                        "roi": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        },
                        "spatialCoordinates": {
                            "x": int(t.spatialCoordinates.x),
                            "y": int(t.spatialCoordinates.y),
                            "z": int(t.spatialCoordinates.z)
                        }
                    }
                    objects.append(tracklet_data)
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)


                yield cv2.imencode('.jpg', frame)[1].tobytes(), objects, output


 