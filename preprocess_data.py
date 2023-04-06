import readline
from struct import Struct
import pyrealsense2.pyrealsense2 as rs
from PIL import Image
from glob import glob
import os

def to_binary(color_img, throttle, steering, heading, index, filename = None):
    if (filename is None):
        #ensure the data subdirectory exists
        filename = f"data/data_{index}_{time.time()}.bin"
    file = open(filename, "wb")

    #A binary file was used instead of a bag file
    file.write(color_img.tobytes())
    file.write(np.array([throttle, steering, heading, index]).tobytes())
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

output_dir = "" # PLACE DATA OUTPUT LOCATION HERE
def main():
    os.chdir("") # PLACE DATA INPUT (.bag) LOCATION HERE
    files = glob("*.bag", recursive=False)
    for bag_file in files:
        gen_file_name = bag_file.rstrip(".bag")
        csv = gen_file_name + ".csv"
        pipeline = create_pipeline(bag_file)
        csv_fp = open(csv)
        line = None
        index = 0

        while line != "":
            lines = []
            subdir = f"{index}_{gen_file_name}_dir"
            os.mkdir(os.path.join(output_dir, subdir))
            for i in range(5):
                line = csv_fp.readline()
                if (line == ""):
                    break
                lines.append(line)

            if (line == ""):
                continue

            for i in range(5):
                frame  = pipeline.wait_for_frames().get_color_frame()
                while(frame.frame_number):
                    frame  = pipeline.wait_for_frames().get_color_frame()

                throttle, steering, heading, index = get_attributes(lines[i])
                to_binary(frame, throttle, steering, heading, index, os.path.join(output_dir, subdir, f'{i}.bin')) 
            index += 1





                
