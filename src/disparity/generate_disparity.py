import os
import cv2
import argparse
import numpy as np
import itertools
import shutil
import time
from pathlib import Path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import PIL.Image as Image
from concurrent.futures import ThreadPoolExecutor

# GIF
import glob
from PIL import Image


def print_elapsed_time(prefix=""):
    e_time = time.time()
    if not hasattr(print_elapsed_time, "s_time"):
        print_elapsed_time.s_time = e_time
    else:
        print(f"{prefix} elapsed time: {e_time - print_elapsed_time.s_time:.2f} sec")
        print_elapsed_time.s_time = e_time


def normalize(image):
    return cv2.normalize(
        src=image, dst=image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX
    )


# TODO: return confidence map
def process_frame_BM_postproc(
    left,
    right,
    name,
    output_dir,
    do_blur=True,
    do_downsample=True,
    do_plot=False,
    **kwargs,
):
    numDisparities = kwargs.get("disparity")

    kernel_size = 3
    left = cv2.GaussianBlur(left, (kernel_size, kernel_size), 1.5)
    right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

    left_matcher = cv2.StereoBM_create(
        numDisparities=kwargs.get("disparity"), blockSize=kwargs.get("blockSize")
    )
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter.setLambda(kwargs.get("lambdas"))
    wls_filter.setSigmaColor(kwargs.get("sigmaColors"))

    disparity_left = np.int16(left_matcher.compute(left, right))
    disparity_right = np.int16(right_matcher.compute(right, left))

    wls_image = wls_filter.filter(
        disparity_map_left=disparity_left,
        left_view=left,
        right_view=right,
        disparity_map_right=disparity_right,
    )

    wls_image = normalize(wls_image)
    wls_image = np.uint8(wls_image)

    # crop - https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    _, thresh = cv2.threshold(wls_image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = wls_image[y : y + h, x : x + w]

    cv2.imwrite(os.path.join(output_dir, name + ".png"), crop)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def process_dataset(left_dir, right_dir, output_dir, algo="BM_POST", **kwargs):
    left_images = [f for f in listdir(left_dir) if f.endswith(".png")]
    right_images = [f for f in listdir(right_dir) if f.endswith(".png")]
    assert len(left_images) == len(right_images)
    left_images.sort()
    right_images.sort()

    executor = ThreadPoolExecutor()
    for params in product_dict(**kwargs):
        for i in range(len(left_images)):
            left_image_path = os.path.join(left_dir, left_images[i])
            right_image_path = os.path.join(right_dir, right_images[i])
            imgL = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

            executor.submit(
                process_frame_BM_postproc,
                imgL,
                imgR,
                left_images[i].split(".")[0],
                output_dir=output_dir,
                **params,
            )
        # createGifs()
    executor.shutdown(wait=True)


def createGifs(input_dir, output_dir):
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    fp_in = input_dir + "*.png"
    fp_out = f"{glob.glob(fp_in)[0].split('.')[0]}.gif"
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format="GIF", append_images=imgs, save_all=True, loop=0)

    # delete imgs
    for filename in os.listdir(fp_in.split("*")[0]):
        if filename.endswith(".png"):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create disparity maps using StereoBM."
    )
    parser.add_argument(
        "-l",
        "--left_dir",
        dest="left_dir",
        required=True,
        help="directory of left daVinci images",
        metavar="DIR",
    )
    parser.add_argument(
        "-r",
        "--right_dir",
        dest="right_dir",
        required=True,
        help="directory of right daVinci images",
        metavar="DIR",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        required=True,
        help="output directory of daVinci disparity maps",
        metavar="DIR",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # BM params
    NUM_DISPARITIES = [64]
    BLOCK_SIZES = [15]

    # optional processing
    do_downsample = False
    do_blur = True
    if do_downsample:
        NUM_DISPARITIES = [int(x / 2) for x in NUM_DISPARITIES]

    # postprocessing params
    LAMBDAS = [8000]
    SIGMA_COLORS = [0.8]

    print_elapsed_time()
    process_dataset(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        output_dir=args.output_dir,
        disparity=NUM_DISPARITIES,
        blockSize=BLOCK_SIZES,
        lambdas=LAMBDAS,
        sigmaColors=SIGMA_COLORS,
    )
    print_elapsed_time()
