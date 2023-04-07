import tensorflow as tf
import tensorflow.keras as keras
import sys
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
from  utilities import drone_lib as dl
from dronekit import connect, VehicleMode
import time
import sys
import score
from imutils.video import FPS


def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device

def get_model(model_file):
    model = keras.models.load_model(model_file, compile=False)
    model.compile(optimizer='adam', loss='mse')
    return model

def get_predictions(heading, frame):
    return model.predict((np.array([heading]), np.array([frame])))

def get_scored_img(img, throttle):
    scored_img= np.multiply((img / 255), kernel)
    score = np.sum(scored_img)
    return scored_img, score

white_threshold = np.array([215, 215, 215])
white = np.array([255, 255, 255])
def process_img(img, throttle):
    rs_img = cv2.resize(img, (320, 240))
    img_bw = cv2.inRange(rs_img, white_threshold, white)[0:320][80:240]
    score_img, score = get_scored_img(img_bw, throttle)
    img_stacked = np.dstack((img_bw, score_img))
    return img_stacked, score

def run():
    print("Running...")
    frame = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
    p_frame, score = process_img(frame)
    preds = get_predictions(score, p_frame)

    fps = FPS().start()
    while drone.armed:
        frame = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
        p_frame, score = process_img(frame)

        preds = get_predictions(score, p_frame)

        steering = int(preds[0][0])
        throttle = int(preds[0][1])
        steering = 0 if steering < 0 else steering
        throttle = 0 if throttle < 0 else throttle

        drone.channels.overrides = {'1': steering, '3': throttle}
        fps.update()
    fps.stop()
    print(f"{fps.fps()} frames processed per second")
    print("Drone disarmed")




if len(sys.argv) > 1:
    model_file = sys.argv[1]
else:
    model_file = "model"

height = 160
width = 320
model = get_model(model_file)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("Connecting to device")
drone = connect(ip='/dev/ttyUSB0', wait_ready=True)
print("Device connected.")
print(f"Device version: {drone.version}")

kernel = score.get_gaussian_matrix(h=height, w=width, h_stdev=0.6, w_stdev=.04, shift=100)
kernel += score.get_gaussian_matrix(h=height, w=width, h_stdev=0.2, w_stdev=.15, shift=40)
kernel /= 2

while not drone.armed:
    print("Please switch device to armed...")
    drone.mode = VehicleMode("MANUAL")
    drone.armed = True
    time.sleep(1)

run()
