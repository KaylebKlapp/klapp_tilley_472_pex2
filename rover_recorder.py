from xmlrpc.client import DateTime
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import sys
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import logging
from imutils.video import FPS
from  utilities import drone_lib as dl
import random as rand
import pandas as pd
import csv
from datetime import datetime

def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device


def arm_device(device):
    while device.mode != "MANUAL":
        print("Switching to MANUAL  mode...")
        device.mode = VehicleMode("MANUAL")
        time.sleep(2)
    while not device.armed:
        print("Please switch device to armed...")
        time.sleep(1)
        
def stream_video(pipeline):
    while (True):

        frame = pipeline.wait_for_frames().get_color_frame()
        frameformatted = np.asanyarray(frame.get_data())
        cv2.imshow('rgb', frameformatted)

        depth = pipeline.wait_for_frames().get_depth_frame()
        depthformatted = np.asanyarray(depth.get_data())
        cv2.imshow('depth', depthformatted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
def append_ardu_data(throttle, steering, heading, idx ):
    fp = open(csv_file, 'a')
    fp.write(f"{throttle},{steering},{heading},{idx}\n")
    fp.close()



def bind(rover, pipeline, logging, fps):
    state_update_interval = 104
    # stream_video(pipeline)
    while rover.armed:
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            bgr_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not bgr_frame or not depth_frame:
                continue

            if (rover.channels['3'] is None
                or rover.channels['1'] is None):
                print("None found in channel")
                continue

            throttle = int(rover.channels['3'])
            steering_mix = int(rover.channels['1'])
            frm_num = int(bgr_frame.frame_number)
            heading = rover.heading

            # write throttle and steering related to current frame...
            append_ardu_data(throttle=throttle, steering=steering_mix,
                             heading=heading, idx=frm_num)

            if (frm_num % state_update_interval) == 0:
                dl.display_rover_state(rover)

            # update the FPS counter
            fps.update()
            
        except Exception as e:

            logging.error("Unexpected error while streaming.", exc_info=True)
            break

    logging.info("Stopping recording...")


def start():
    log_file = "logger.txt"
    #rover = connect_device("127.0.0.1:14550")
    rover = connect_device('/dev/ttyUSB0')
    arm_device(rover)
    # prepare log file...
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.info(f"Recording to be stored in location: {bag_file}.")
    logging.info("Preparing RealSense data streams...")
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_record_to_file(f"{bag_file}")
    logging.info("Configuring depth stream.")
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    logging.info("configuring rgb stream.")
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    logging.info("Starting camera streams...")
    pipeline.start(config)
    # for now, we do not need direct access to device or recorder...
    # device = pipeline.get_active_profile().get_device()
    # recorder = rs.recorder(device)
    fps = FPS().start()
    logging.info("Recording for realsense sensor streams started.")


    return logging, pipeline, fps, rover


def main(args = None):
    log, pipe, fps, rover = start()
    bind(rover, pipe, log, fps)
    pipe.stop()

if __name__ == "__main__":
    prefix = "" if len(sys.argv) == 1 else sys.argv[1] + "_"
    run = True
    while (run):
        collect_key = datetime.now().strftime('%m%d_%H%M_%S')
        csv_file = f'/media/usafa/data/telemetry/csvs/{prefix}{collect_key}.csv'
        bag_file = f"/media/usafa/data/telemetry/{prefix}{collect_key}.bag"

        # execute only if run as a script
        main()
        print("Pausing for realsense buffer...")
        time.sleep(10)
        again = input("Run another collection? (y/n) ")
        run = (again == 'y')
        if (run):
            prefix = input("File prefix: ") + "_"
            
