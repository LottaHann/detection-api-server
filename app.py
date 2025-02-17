#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response, jsonify
import json
import time
from flask_cors import CORS



# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

laptop_ip = "193.167.36.67"
rpi_ip = "193.166.180.103"
expression_port = 8008
main_server_port = 5100
main_frontend_port = 5500
#etection_server_url = os.environ.get('SERVER_URL', 'http://127.0.0.1:8008')
expression_server_url = f'http://{rpi_ip}:{expression_port}'
main_server_url = f'http://{laptop_ip}:{main_server_port}'
main_frontend_url = f'http://{laptop_ip}:{main_frontend_port}'

app = Flask(__name__)
CORS(app, resources={r"/see": {"origins":[ expression_server_url]}})




@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame, objects, cameraDebug= camera.get_frame()
        print(f"\033[2J\033[1;1H{cameraDebug}\n{objects}", end="", flush=True,)
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'

def jsonData(camera):
    """Jsondata streaming generator function."""
    debug = False
    # yield b'--frame\r\n'
    while True:
        testData  =[ {
                        "id": "0",
                        "label": "person",
                        "status": "TRACKED",
                        "roi": {
                            "x1": "0",
                            "y1": "0",
                            "x2": "0",
                            "y2": "0"
                        },
                        "spatialCoordinates": {
                            "x": int("10"),
                            "y": int("10"),
                            "z": int("400")
                        }
                    }]
        if(not debug):
            frame, objects, cameraDebug= camera.get_frame()
            json_data = objects
            print(f"\033[2J\033[1;1H{cameraDebug}\n{objects}", end="", flush=True,)
            send_data =json.dumps(json_data)
            return send_data

        if(debug):
            return json.dump(testData)
        
        # return json.dumps(objects)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/see")
def data():
    """Json data streaming route. Use this data in other client side applications."""
    return Response(jsonData(Camera()), mimetype='application/json')

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8008, threaded=True)
