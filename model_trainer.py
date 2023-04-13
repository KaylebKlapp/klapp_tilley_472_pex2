import random

import pyrealsense2.pyrealsense2 as rs
import sklearn.model_selection
from cv2 import resize, inRange, imshow, waitKey
from os import chdir, rename, path, getcwd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
import matplotlib.pyplot as plt
import score as sc
from random import sample, randint
height = 160
width = 320


def build_model():
    inputs = layers.Input(shape=(1,), name='inputs')
    images = layers.Input(shape=(height, width, 2), name='images')
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(images)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    flatten = layers.Flatten()(pool3)
    concat = layers.Concatenate()([flatten, inputs])
    dense5 = layers.Dense(256, activation='relu')(concat)
    dense4 = layers.Dense(128, activation='relu')(dense5)
    dense3 = layers.Dense(64, activation='relu')(dense4)
    dense2 = layers.Dense(32, activation='relu')(dense3)
    dense1 = layers.Dense(16, activation='relu')(dense2)
    outputs = layers.Dense(2, name='outputs')(dense1)

    model = keras.Model(inputs=[inputs, images], outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[keras.losses.mean_absolute_percentage_error])
    model.summary()
    return model


def get_attributes(csv_string):
    csv_string = csv_string.rstrip()
    throttle, steering, heading, index = [int(val) for val in csv_string.split(",")]
    return throttle, steering, heading, index


white_threshold = np.array([205, 205, 205])
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


def process_img(img, throttle):
    rs_img = resize(img, (320, 240))
    img_bw = inRange(rs_img, white_threshold, white)[0:width][80:240]
    score_img, score = get_scored_img(img_bw, throttle)
    img_stacked = np.dstack((img_bw, score_img))

    return img_stacked, score


num_bytes_img = 921600


def unbin_file(file):
    fp = open(file, 'rb')
    img = np.frombuffer(fp.read(num_bytes_img), np.uint8).reshape(480, 640, 3)
    throttle, old_throttle, steering, heading, old_heading, index = np.frombuffer(fp.read(48), np.uint64).reshape(6).tolist()
    fp.close()

    return img, throttle, old_throttle, steering, heading, old_heading, index


path_to_data = "/home/usafa/Desktop/team_just_kidding/experimental_processed_data"
#path_to_data = '/media/usafa/ext_data/experimental_processed_data_flipped'

def generate_training_data_shuffled(batch_size = 500, files=None):
    sub_dirs = glob("*_dir", recursive=False) if files is None else files
    random.shuffle(sub_dirs)
    while True:
        for file_dir in sub_dirs:
            if ("noise" in file_dir):
                continue

            inputs = []
            frames = []
            outputs = []
            for i in range(int(batch_size / 5)):
                try:
                    for dat_file in glob(file_dir + "/*", recursive=False):
                        img, throttle, old_throttle, steering, heading, old_heading, index = unbin_file(dat_file)
                        if (steering > 1500 or steering < 1490):
                            loop = 5
                            i += 4
                        else:
                            loop = 1
                        for i in range(loop):
                            p_img, score = process_img(img, old_throttle)
                            frames.append(p_img)
                            outputs.append([steering, throttle])
                            inputs.append(score)
                except:
                    print(f"Bad file data found at ", dat_file)
            if (len(outputs) != 0):
                yield [np.array(inputs), np.array(frames)], np.array(outputs)


def generate_validation_data_preprocessed(files):
    dat_files = []
    for file in files:
        dat_files += (glob(f"{file}/*.bin"))

    frames = []
    outputs = []
    inputs = []
    old_heading = -1
    index = 0
    for dat_file in dat_files:
        img, throttle, old_throttle, steering, heading, old_heading, index = unbin_file(dat_file)
        p_img, score = process_img(img, old_throttle)
        frames.append(p_img)
        outputs.append([steering, throttle])
        inputs.append(score)
        
    return [np.array(inputs), np.array(frames)], np.array(outputs)


def split_training_validation_data(batch_size, num_validation=400):
    files = glob("*_dir", recursive=False)
    files_per_dir = len(glob(files[0] + "/*.bin", recursive=False))
    validation_files = random.sample(files, int(num_validation / files_per_dir))
    for fp in validation_files:
        files.remove(fp)
    return (files_per_dir * len(files)) / batch_size, files, validation_files


def get_image_score(img, throttle):
    img_score, _, _ = sc.score_state(kernel, img, throttle / 2000)
    return img_score

def get_scored_img(img, throttle):
    scored_img= np.multiply((img / 255), kernel)
    score = np.sum(scored_img) * (throttle / 2000)
    return scored_img, score

model_version = 15
def train_model():
    batch_size = 64
    epochs = 40
    step_num, train_files, validation_files = split_training_validation_data(batch_size)
    validation_data, validation_y = generate_validation_data_preprocessed(validation_files)
    x_dat = generate_training_data_shuffled(batch_size=batch_size, files=train_files)
    device = '/GPU:0'
    history = None
    with tf.device(device):
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

    return model


curdir = getcwd()
chdir("/home/usafa/Desktop/team_just_kidding/experimental_processed_data")
model = None
kernel = sc.get_gaussian_matrix(h=height, w=width, h_stdev=1, w_stdev=.18, shift=40)
kernel += sc.get_gaussian_matrix(h=height, w=width, h_stdev=0.2, w_stdev=.4, shift=100)
kernel /= 2
model = train_model()
# imshow('kernel', kernel)
# waitKey(0)



