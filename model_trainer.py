import pyrealsense2.pyrealsense2 as rs
from cv2 import resize, inRange, imshow, waitKey
from os import chdir
import numpy as np


def get_attributes(csv_string):
    csv_string = csv_string.rstrip()
    heading, steering, throttle, index = [int(val) for val in csv_string.split(",")]
    return heading, steering, throttle, index

white_threshold = np.array([215, 215, 215])
white = np.array([255, 255, 255])
def process_img(img):
    rs_img = resize(img, (320, 240))
    img_bw = inRange(rs_img, white_threshold, white)
    return  img_bw[0:320][80:240]

def process_train_file(file):
    bag_file = file + ".bag"
    csv_file =  "csvs/" + file + ".csv"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    csv_fp = open(csv_file, "r")
    line = csv_fp.readline()
    while line != "":
        count = 0
        inputs = []
        frames = []
        outputs = []
        indexes = []
        old_heading = 0
        while count < 100 and line != "":
            heading, steering, throttle, index = get_attributes(line)
            inputs.append([heading - old_heading])
            old_heading = heading
            outputs.append([steering, throttle])
            line = csv_fp.readline()
            indexes.append(index)
            count += 1

        for i in range(count):
            frame = pipeline.wait_for_frames().get_color_frame()
            while frame.frame_number < indexes[i]:
                frame = pipeline.wait_for_frames().get_color_frame()

            img = np.asanyarray(frame.get_data())
            frames.append(process_img(img))

            
            # imshow("img", process_img(img))
            # waitKey(1)

    csv_fp.close()
    pipeline.stop()


        #Write code to train model 100 items at a time
chdir("/media/usafa/extern_data/Team Just Kidding/Collections/")
process_train_file("c_1.1_0313_1040_42")



