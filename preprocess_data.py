import readline
from struct import Struct
import pyrealsense2.pyrealsense2 as rs
from PIL import Image
from glob import glob
import os
from time import time
import numpy as np


def to_binary(color_img, throttle, steering, heading, old_heading, index, filename = None):
    if filename is None:
        filename = f"data/data_{index}_{time()}.bin"
    file = open(filename, "wb")

    file.write(color_img.tobytes())
    file.write(np.array([throttle, steering, heading, old_heading, index]).tobytes())
    file.close()


def create_pipeline(file):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    return pipeline


def get_attributes(csv_string):
    csv_string = csv_string.rstrip()
    throttle, steering, heading, index = [int(val) for val in csv_string.split(",")]

    return throttle, steering, heading, index


# PLACE DATA OUTPUT LOCATION HERE
output_dir = "/home/usafa/Desktop/team_just_kidding/processed_data"


def main():
    # PLACE DATA INPUT (.bag) LOCATION HERE
    os.chdir("/home/usafa/Desktop/team_just_kidding/collections/")
    files = glob("*.bag", recursive=False)
    for bag_file in files:
        gen_file_name = bag_file.rstrip(".bag")
        csv = "csvs/" + gen_file_name + ".csv"
        pipeline = create_pipeline(bag_file)
        csv_fp = open(csv)
        line = None
        dir_index = 0

        while line != "":
            lines = []
            subdir = f"{gen_file_name}_{dir_index}_dir"
            for i in range(5):
                line = csv_fp.readline()
                if line == "":
                    break
                lines.append(line)

            if line == "":
                continue

            os.mkdir(os.path.join(output_dir, subdir))
            old_heading = -1
            for i in range(5):
                throttle, steering, heading, index = get_attributes(lines[i])
                if old_heading == -1:
                    old_heading = heading
                frame = pipeline.wait_for_frames().get_color_frame()
                while frame.frame_number < index:
                    frame = pipeline.wait_for_frames().get_color_frame()
                img = np.asanyarray(frame.get_data())

                to_binary(img, throttle, steering, heading, old_heading, index, os.path.join(output_dir, subdir, f'{i}.bin'))
                if dir_index == 10:
                    print(f"{throttle}, {steering}, {heading}, {old_heading}, {index}")

                old_heading = heading
            dir_index += 1
        print(f"Finished file {bag_file}")

main()





                
