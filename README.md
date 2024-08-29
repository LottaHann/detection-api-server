flask-video-streaming
=====================

Supporting code for Miguel Grinbergs article [video streaming with Flask](http://blog.miguelgrinberg.com/post/video-streaming-with-flask) and its follow-up [Flask Video Streaming Revisited](http://blog.miguelgrinberg.com/post/flask-video-streaming-revisited).

This project is a proof of concept to use the luxonis cameras with a video streaming server.

To run:
Build the docker container

docker build -t detection-server .

Start the container: 
docker run -d -p 8008:8008 -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' detection-server

Now go to localhost:8008 to see the video feed from the camera.


