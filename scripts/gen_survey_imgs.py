import argparse
import os
import random
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd

IMAGE_WIDTH = 384
IMAGE_HEIGHT = 192
CANVAS_SIZE = (880, 600)
QUESTION = "Given the left eye view, which of the generated right eye views (A or B) is better quality?"


def get_image(base_path, exp_num_or_name, frame_num):
    exp_folder = None

    if exp_num_or_name == "target" or exp_num_or_name == "targets":
        path = os.path.join(args.exp_dir, "targets")
        image = Image.open(os.path.join(path, f"target_{frame_num}.png"))
    elif exp_num_or_name == "input" or exp_num_or_name == "inputs":
        path = os.path.join(args.exp_dir, "inputs", f"{frame_num}")
        fname = sorted(os.listdir(path))[-1]
        old_img = Image.open(os.path.join(path, fname))

        old_size = old_img.size

        new_size = (IMAGE_WIDTH+30, IMAGE_HEIGHT+30)
        new_img = Image.new('RGB', new_size, color=(200, 200, 200, 0))
        new_img.paste(old_img, ((new_size[0]-old_size[0])//2,
                                (new_size[1]-old_size[1])//2))

        image = new_img
    else:
        # Get folder of experiment
        for fname in os.listdir(args.exp_dir):
            path = os.path.join(args.exp_dir, fname)
            if os.path.isdir(path):
                if exp_num_or_name == fname.split("-")[0]:
                    exp_folder = fname
                    break

        path = os.path.join(args.exp_dir, exp_folder, "version_0")

        epoch_dirs = [fname for fname in os.listdir(path) if
                      os.path.isdir(path) and "epoch" in fname]

        max_epoch = max(int(x[6:]) for x in epoch_dirs)
        max_epoch_folder_name = "epoch_" + str(max_epoch)

        image = Image.open(os.path.join(path, max_epoch_folder_name, f"pred_{frame_num}.png"))

    return image


def create_comp_img(img_input, img_a, img_b, q_num):
    canvas = Image.new('RGB', CANVAS_SIZE, color=(255, 255, 255, 0))

    # Create question text
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", size=20)
    d.text((15, 25), f"Q{q_num}: " + QUESTION, font=font, fill="black")

    canvas.paste(img_a, (CANVAS_SIZE[0] - IMAGE_WIDTH - 20, 90))
    d.text((CANVAS_SIZE[0] - int(IMAGE_WIDTH/2)-25, 90 +
            IMAGE_HEIGHT), "A", font=font, fill="black")

    canvas.paste(img_b, (CANVAS_SIZE[0] - IMAGE_WIDTH - 20, 340))
    d.text((CANVAS_SIZE[0] - int(IMAGE_WIDTH/2)-25, 148 +
            2*IMAGE_HEIGHT), "B", font=font, fill="black")

    canvas.paste(img_input, (20, 200))
    d.text((int(IMAGE_WIDTH/2)-20, int(CANVAS_SIZE[1] / 2) +
            int(IMAGE_HEIGHT/2) + 23), "LEFT VIEW", font=font, fill="black")

    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images for survey')
    parser.add_argument('--exps', type=str,
                        help='CSV file containing list of experiments', required=True)
    parser.add_argument('--out_dir', type=str,
                        help='output directory for images', required=True)
    parser.add_argument("--exp_dir", type=str, required=True, help="path to davinci data")
    parser.add_argument("--padding", type=int, default=3,
                        required=False, help="padding for images")
    args = parser.parse_args()

    # create dict of models to frame numbers
    df = pd.read_csv(args.exps)
    for index, row in df.iterrows():
        # get image model a and model b
        q_num = row['quest_num']
        model_a = row['model_a']
        model_b = row['model_b']
        frame_num = row['frame_num']

        img_a = get_image(args.exp_dir, model_a, frame_num)
        img_b = get_image(args.exp_dir, model_b, frame_num)
        img_input = get_image(args.exp_dir, "input", frame_num)

        # stitch
        output_img = create_comp_img(img_input, img_a, img_b, q_num)
        output_img.save(os.path.join(args.out_dir, f"Q{q_num}.png"))
