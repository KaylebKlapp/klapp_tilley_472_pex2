import sys
import numpy as np
import cv2
import rosbag
import csv
from cv_bridge import CvBridge as bridge
from pandas import *
import pyrealsense2.pyrealsense2 as rs
from PIL import Image
import imageio
from imutils.video import FPS

def organize(fps):
    count = 0

    data = read_csv("rover.csv")
    throttle = data["Steering"].tolist()
    steering = data["Throttle"].tolist()


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file("telemetry.bag")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Streaming loop
    while True:

        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        color_frame = frames.get_color_frame()
        depth_frame  = frames.get_depth_frame()



        # convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        down_width = 320
        down_height = 240
        down_points = (down_width, down_height)
        resized_depth = cv2.resize(depth_image, down_points, interpolation=cv2.INTER_LINEAR)
        resized_color = cv2.resize(color_image, down_points, interpolation=cv2.INTER_LINEAR)

        grey_image  = cv2.cvtColor(resized_color,cv2.COLOR_BGR2GRAY)
        (thresh, bw_image)  = cv2.threshold(grey_image, down_height,down_width, cv2.THRESH_BINARY)
        cropper_bw = bw_image[80:240]

        imageio.imwrite("photos/color: throttle: "+ str(throttle[count]) + "Steering: "+ str(steering[count])+".png",resized_color)
        imageio.imwrite("photos/depth: throttle: " + str(throttle[count]) + "Steering: " + str(steering[count]) + ".png", resized_depth)
        imageio.imwrite( "photos/bw: throttle: " + str(throttle[count]) + "Steering: " + str(steering[count]) + ".png", cropper_bw)

        cv2.imshow("BW", cropper_bw)
        cv2.imshow("COLOR", resized_color)
        cv2.waitKey(0)
        cv2.destroyWindow()

        count +=1
        if count  == len(throttle):
            break

        fps.update()

    return fps
def main(args):
    fps = FPS().start()
    fps = organize(fps)
    fps.stop()
    print("FPS:  {:.2f}".format(fps.fps()))


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
