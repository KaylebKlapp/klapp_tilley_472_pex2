import pyrealsense2.pyrealsense2 as rs
from cv2 import resize, inRange, imshow, waitKey
from os import chdir, rename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
from random import sample
height = 160
width = 320

def build_model():
    headings = layers.Input(shape=(1,), name='headings')
    images = layers.Input(shape=(height, width,1), name='images')
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(images)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    flatten = layers.Flatten()(pool1)
    dense1 = layers.Dense(32, activation='relu')(flatten)
    concat = layers.Concatenate()([dense1, headings])
    dense2 = layers.Dense(16, activation='relu')(concat)
    outputs = layers.Dense(2, name='outputs')(dense2)

    model = keras.Model(inputs=[headings, images], outputs=outputs)

    c_model = model.compile(optimizer='adam', loss={'outputs': 'mse'},
                  metrics={'throttle': 'mae', 'steering': 'mae'})
    
    model.save("model")
    return model


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

def generate_training_data(num_samples = 512):
    print("Hello?")
    files = glob("*.bag", recursive= False)

    for file_name in files:
        if ("TRAINED" in file_name):
            continue

        generic_file_name = file_name.rstrip(".bag")
        bag_file = file_name
        csv_file =  "csvs/" + generic_file_name + ".csv"

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        csv_fp = open(csv_file, "r")
        line = csv_fp.readline()
        first_run = True
        old_heading = None
        while line != "":
            if first_run:
                inputs = []
                frames = []
                outputs = []
                indexes = []
                old_heading = 0
                count = 0
                first_run = False
            else:
                inputs = inputs[int(count / 2): count]
                frames = frames[int(count / 2): count]
                outputs =outputs[int(count / 2): count]
                indexes = indexes[int(count / 2): count]
                count = int(count/2)

            while count < num_samples and line != "":
                heading, steering, throttle, index = get_attributes(line)
                line = csv_fp.readline()
                indexes.append(index)
                
                frame = pipeline.wait_for_frames().get_color_frame()
                while frame.frame_number < indexes[count]:
                    frame = pipeline.wait_for_frames().get_color_frame()

                img = np.asanyarray(frame.get_data())
                inputs.append([heading - old_heading])
                outputs.append([steering, throttle])
                frames.append(process_img(img))
                
                old_heading = heading
                count += 1

            for frame in frames:
                imshow("hello", frame)
                waitKey(10)
            yield  [frames, inputs], outputs

        csv_fp.close()
        pipeline.stop()
        #rename(csv_file, f"TRAINED_{csv_file}")
        #rename(csv_file, f"TRAINED_{bag_file}")

def generate_validation_data(file_path, num_samples=512):
    bag_file = file_path
    csv_file =  "csvs/" + bag_file.replace(".bag", ".csv")
    csv_fp = open(csv_file, "r")
    lines = csv_fp.readlines()
    num_lines = len(lines)
    offset = int(num_lines / num_samples)

    validation_frames = [] 
    validation_inputs = [] 
    validation_outputs = [] 

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    old_heading = 0
    for i in range(0, num_lines, offset):
        old_heading, _, _ = get_attributes(lines[i - 1]) if i is not 1 else 0
        heading, steering, throttle, index = get_attributes(lines[i])
        frame = pipeline.wait_for_frames().get_color_frame()
        while frame.frame_number < index:
            frame = pipeline.wait_for_frames().get_color_frame()
        old_heading = old_heading - heading

        validation_inputs.append([heading - old_heading])
        validation_frames.append(process_img(np.asanyarray(frame.get_data())))
        validation_inputs.append(heading)
        validation_inputs.append([steering, throttle])


    return [validation_frames, validation_inputs], validation_outputs


#validation_data = generate_validation_data(validation_fp)

def train_model(model, batch_size = 32):
    # history = model.fit(
    #     generate_training_data,
    #     validation_data=validation_data,
    pass




#chdir("/media/usafa/extern_data/Team Just Kidding/Collections/")
model = build_model()




