import pyrealsense2.pyrealsense2 as rs
from cv2 import resize, inRange, imshow, waitKey
from os import chdir, rename, path, getcwd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
import matplotlib.pyplot as plt
from random import sample
height = 160
width = 320


def build_model():
    headings = layers.Input(shape=(1,), name='headings')
    images = layers.Input(shape=(height, width,1), name='images')
    conv1 = layers.Conv2D(32, (6, 6), activation='relu')(images)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv2)
    flatten = layers.Flatten()(pool1)
    dense1 = layers.Dense(32, activation='relu')(flatten)
    concat = layers.Concatenate()([dense1, headings])
    dense2 = layers.Dense(64, activation='relu')(concat)
    dense4 = layers.Dense(32, activation='relu')(dense2)
    dense5 = layers.Dense(16, activation='relu')(dense4)
    outputs = layers.Dense(2, name='outputs')(dense5)

    model = keras.Model(inputs=[headings, images], outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[keras.losses.mean_absolute_percentage_error])
    model.summary()
    return model


def get_attributes(csv_string):
    csv_string = csv_string.rstrip()
    throttle, steering, heading, index = [int(val) for val in csv_string.split(",")]
    return throttle, steering, heading, index


white_threshold = np.array([215, 215, 215])
white = np.array([255, 255, 255])


def create_pipeline(file):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(file)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    return pipeline


def process_img(img):
    rs_img = resize(img, (320, 240))
    img_bw = inRange(rs_img, white_threshold, white)
    return  img_bw[0:320][80:240]


def generate_training_data(batch_size = 512):
    files = glob("*.bag", recursive= False)
    while True:
        for file_name in files:
            if "TRAINED" in file_name or "VALIDATION" in file_name:
                continue

            generic_file_name = file_name.rstrip(".bag")
            bag_file = file_name
            csv_file = "csvs/" + generic_file_name + ".csv"

            pipeline = create_pipeline(bag_file)

            csv_fp = open(csv_file, "r")
            line = csv_fp.readline()
            first_run = True
            old_heading = 0

            while line != "":
                if True:
                    inputs = []
                    frames = []
                    outputs = []
                    indexes = []
                    old_heading = 0
                    count = 0
                    first_run = False
                # else:
                #     inputs = inputs[int(count / 2): count]
                #     frames = frames[int(count / 2): count]
                #     outputs =outputs[int(count / 2): count]
                #     indexes = indexes[int(count / 2): count]
                #     count = int(count/2)

                for i in range(batch_size):
                    line = csv_fp.readline()
                    if line == "":
                        break

                    throttle, steering, heading, index = get_attributes(line)
                    indexes.append(index)

                    frame = pipeline.wait_for_frames().get_color_frame()
                    while frame.frame_number < indexes[count]:
                        frame = pipeline.wait_for_frames().get_color_frame()

                    img = np.asanyarray(frame.get_data())
                    inputs.append(heading - old_heading)
                    outputs.append(np.array([steering, throttle]))
                    frames.append(process_img(img))

                    old_heading = heading
                    count += 1
                if count == batch_size:
                    yield [np.array(inputs), np.array(frames)], np.array(outputs)
            csv_fp.close()
            pipeline.stop()
            #rename(csv_file, f"TRAINED_{csv_file}")
            #rename(csv_file, f"TRAINED_{bag_file}")

def generate_validation_data(num_samples = 1024):
    num_samples = num_samples/2
    validation_frames = []
    validation_inputs = []
    validation_outputs = []
    for file in glob("*VALIDATION*", recursive= False):
        bag_file = file
        csv_file = bag_file.rstrip(".bag")
        csv_file = "csvs/" + csv_file + ".csv"
        csv_fp = open(csv_file, "r")
        lines = csv_fp.readlines()
        num_lines = len(lines)
        offset = int(num_lines / num_samples)

        pipeline = create_pipeline(bag_file)
        for i in range(0, num_lines, offset):
            _, _, old_heading, _ = get_attributes(lines[i - 1]) if i != 1 else [0,0,0,0]
            throttle, steering, heading, index = get_attributes(lines[i])
            frame = pipeline.wait_for_frames().get_color_frame()

            while frame.frame_number < index:
                frame = pipeline.wait_for_frames().get_color_frame()

            validation_inputs.append(heading - old_heading)
            validation_frames.append(process_img(np.asanyarray(frame.get_data())))
            validation_outputs.append(np.array([steering, throttle]))

    return [np.array(validation_inputs), np.array(validation_frames)], np.array(validation_outputs)


def get_samples_length():
    files = glob("*.bag", recursive= False)
    length = 0
    for file_name in files:
        if "TRAINED" in file_name or "VALIDATION" in file_name:
            continue

        generic_file_name = "csvs/" + file_name.rstrip(".bag") + ".csv"
        fp = open(generic_file_name, 'r')
        length += len(fp.readlines())
    return length


model_version = 2
def train_model(model):
    batch_size = 32
    epochs = 20
    x_dat = generate_training_data(batch_size=batch_size)
    step_num = get_samples_length() / batch_size
    DEVICE = '/GPU:0'
    with tf.device(DEVICE):
        model = build_model()
        save_fp = f'{curdir}/Models/model_{model_version}_'+'{epoch:04d}_{val_loss:.3f}.hdf5'
        save_best = keras.callbacks.ModelCheckpoint(filepath=save_fp, monitor='val_loss', save_best_only=True)
        history = model.fit(
            x=x_dat,
            validation_data=(validation_data, validation_y),
            batch_size=batch_size,
            steps_per_epoch=step_num,
            epochs=epochs,
            verbose=1,
            callbacks=[save_best])
    chdir(curdir)
    print(f"Saving model to {curdir}")
    model.save("trained_model")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'])
    plt.show()


curdir = getcwd()

if path.isdir("/home/usafa/Desktop/team_just_kidding/collections"):
    chdir("/home/usafa/Desktop/team_just_kidding/collections")
elif path.isdir("/media/usafa/extern_data/Team Just Kidding/Collections/"):
    chdir("/media/usafa/extern_data/Team Just Kidding/Collections/")
elif path.isdir("C:\\Users\\Kayleb\\source\\repos\\472\\Collections"):
    chdir("C:\\Users\\Kayleb\\source\\repos\\472\\Collections")
else:
    chdir("C:/Users/C23Jason.Tilley/Desktop/AI/Collections")


validation_data, validation_y = generate_validation_data()
print(len(validation_data[0]))
model = None
train_model(model)



