import os
import glob
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from sklearn.utils import shuffle
import cv2


def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


def load_images(sample_series,
                batch_size=None,
                num_samples=None,
                offset=0,
                normalize_labels=True):

    if batch_size is None:
        batch_size = len(sample_series)

    if num_samples is None or num_samples <= 0:
        num_samples = len(sample_series)

    batch_series = sample_series[offset:offset + batch_size]
    # Sanity check
    # print(f" {sample_name} data range: {offset}:{offset + batch_size}")
    images = []
    labels = []

    for sample in batch_series:  # For every series...

        # We are taking 2 images,
        # first one is BGR, second is depth sensor.
        try:
            file_name = os.path.basename(sample).replace(".png", "")
            f_num, throttle, steering, f_type = file_name.split('_')

            throttle = int(throttle)
            steering = int(steering)

            # As we stream frames from the realsense camera, we opted to
            # configure that stream to give us color images in BGR order,
            # since opencv uses it natively.
            # So, we will keep this ordering here by not opting to have opencv
            # convert to RGB after reading the image.
            bgr_image = cv2.imread(sample)
            depth_image = cv2.imread(sample.replace("_c.png", "_d.png"))  # , cv2.IMREAD_GRAYSCALE)

            images.append([bgr_image, depth_image])

            if normalize_labels:
                steering = min_max_norm(steering)
                throttle = min_max_norm(throttle)

            labels.append([steering, throttle])

        except Exception as e:
            print(f" [EXCEPTION ENCOUNTERED: {e}; skipping series with sample {sample}.] ")
            # Skip entire series, since all series need to be of the same size
            break

    x_train = np.array(images)
    y_train = np.array(labels)

    # Here we do not hold the values of X_train and y_train,
    # instead we yield the values.
    return x_train, y_train


def get_img_list(root_folder,
                 random_state=None,
                 do_shuffle=True,
                 ends_with="_w.png"):
    samples = []  # simple array to append all the entries present in the subfolders

    # Go through files and collect file names
    sub_folders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

    for folder in sub_folders:
        path = os.path.join(folder, "*.png")
        files = glob.glob(path)
        for file in files:
            # only list main RGB images
            if file.endswith(ends_with):
                samples.append(file)
    if do_shuffle:
        samples = shuffle(samples, random_state=random_state)  # shuffling the total images

    return samples


def get_gaussian_matrix(h=160, w=320, h_stdev=.15, w_stdev=.15,shift = 75):
    '''returns a 2D gaussian matrix with '''

    # See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.gaussian.html
    h += shift
    k1d = signal.gaussian(h, std=h_stdev * h).reshape(h, 1)[0:h-shift]
    k2d = signal.gaussian(w, std=w_stdev * w).reshape(w, 1)

    # Note: The inner product (or dot product) of 2 vectors uT*v would result in a matrix
    #          the size of the outer dimensions of the 2 vectors (i.e., a scalar)

    #       However, the outer product of 2 nxm vectors u*vT would result in a
    #         matrix the size of nxm.
    kernel = np.outer(k1d, k2d)

    return kernel


def score_images_test(size=(160, 320),
                      path="/Users/chad/Documents/GitHub/rover_drone_ai/source/rover/images"):

    kernel = get_gaussian_matrix(size[0], size[1])
    kernel_sum = np.sum(kernel)
    # show matrix
    plt.imshow(kernel)
    plt.show()

    img_list = get_img_list(path)
    images, labels = load_images(img_list, normalize_labels=False)

    for image, label in zip(images,labels):
        pict = image[1]
        if pict is not None:

            pict = pict[:, :, 0]
            # get weighted score on throttle
            score, weighted_img, org_score = score_state(pict, kernel, coef=label[1])

            cv2.putText(image[0], f" s: {score}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=(255, 255, 0))

            cv2.putText(image[0], f"os: {org_score}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=(255, 255, 0))

            cv2.putText(image[0], f"throttle: {label[1]}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=(0, 0, 255))

            # show original image
            plot_matrix(image[0], f"Image Score = {score}")

            # show original heatmap
            plot_matrix(pict, f"Heatmap Score = {org_score}")

            # show scored image
            plot_matrix(weighted_img, f"Matrix Score = {score}")
        else:
            print("Image is None; skipping...")


def score_state(state_image, kernel, coef=1000):
    weighted_img = state_image * kernel
    max_score = state_image.shape[0] * state_image.shape[1] * 255
    score = int(min_max_norm(np.sum(weighted_img), v_min=0, v_max=max_score) * coef)
    org_score = int(min_max_norm(np.sum(state_image), v_min=0, v_max=max_score) * coef)

    return score, weighted_img, org_score


def plot_matrix(matrix, title, plot_image=False):
    # creating a plot
    pixel_plot = plt.figure()

    #pixel_plot.add_axes([0,matrix.shape[1],matrix.shape[0], matrix.shape[1]])
    plt.title(title)

    if plot_image:
        pixel_plot = plt.imshow(matrix)
    else:
        pixel_plot = plt.imshow(
            matrix, cmap='Reds', interpolation='nearest')

    plt.colorbar(pixel_plot)

    # show plot
    plt.show()


if __name__ == "__main__":
    # TEST STUFF
    score_images_test()