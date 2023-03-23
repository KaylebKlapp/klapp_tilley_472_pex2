import tensorflow as tf
import tensorflow.keras as keras
import sys
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
from  utilities import drone_lib as dl
from dronekit import connect, VehicleMode
import time

def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device

def get_model(model_file):
    model = keras.models.load_model(model_file)
    return model

def get_predictions(heading, frame):
    heading = tf.expand_dims(heading, 0)
    frame = tf.expand_dims(frame, 0)
    return model.predict([frame, heading])


white_threshold = np.array([215, 215, 215])
white = np.array([255, 255, 255])
def process_img(img):
    rs_img = cv2.resize(img, (320, 240))
    img_bw = cv2.inRange(rs_img, white_threshold, white)
    return  img_bw[0:320][80:240]

def run():
    print("Running...")
    old_heading = 0
    while drone.is_armed:
        frame = pipeline.wait_for_frames().get_color_frame().get_data()
        p_frame = process_img(frame)
        heading = drone.heading - old_heading
        old_heading = drone.heading

        steering, throttle = get_predictions(p_frame, heading)
        drone.channels['1'] = steering
        drone.channels['3'] = throttle
    print("Drone disarmed")




if len(sys.argv) > 1:
    model_file = sys.argv[1]
else:
    model_file = "model"

model = get_model(model_file)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("Connecting to device")
drone = connect(ip='/dev/ttyUSB0', wait_ready=True)
print("Device connected.")
print(f"Device version: {drone.version}")


while not drone.armed:
    print("Please switch device to armed...")
    time.sleep(1)

run()
