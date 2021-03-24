import argparse
import os
import random
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

IMAGE_WIDTH = 384
IMAGE_HEIGHT = 192


def concat_images(image_paths, size, shape, exp_paths, padding=3, include_inputs_and_targets=False, imgs=None):
    # Open images and resize them
    width, height = size
    images = map(Image.open, image_paths)
    images = [ImageOps.fit(image, size)
              for image in images]

    num_x = shape[0]
    num_y = shape[1]

    if include_inputs_and_targets:
        exp_paths = ["inputs", "targets"] + exp_paths
        num_y += 2

    if imgs:
        num_x = len(imgs) + 1

        # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * num_x) + (padding * (num_x -
                                               1)), (height * num_y) + (padding * (num_y - 1))
    image = Image.new('RGB', image_size)

    # Paste images into final image
    for row, name in zip(range(num_y), exp_paths):

        # Draw text label
        img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", size=50)

        if include_inputs_and_targets:
            if row == 0:
                msg = "input"
            elif row == 1:
                msg = "target"
            else:
                msg = name.split('/')[-2]
        else:
            msg = name.split('/')[-2]

        w, h = d.textsize(msg)
        d.text((5, IMAGE_HEIGHT/2 - 20), msg, font=font, fill="black")
        image.paste(img, (0, (height + padding) * row))

        # Paste images
        for col in range(1, num_x):
            offset = (width+padding) * col, (height + padding) * row
            idx = row * (num_x-1) + col
            image.paste(images[idx-1], offset)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize experiments.')
    parser.add_argument('--exps', metavar='N', type=str, nargs='+',
                        help='experiment names', required=True)
    parser.add_argument("--exp_dir", type=str, required=True, help="path to davinci data")
    parser.add_argument("--padding", type=int, default=3, required=False, help="padding for images")
    parser.add_argument("--inputs_and_targets", action='store_true',
                        help="Include inputs and targets")
    parser.add_argument("--imgs", type=int, nargs='+', default=[],
                        help="List of images to visualize")
    args = parser.parse_args()

    set_exps = set(args.exps)
    exp_paths = []

    for fname in os.listdir(args.exp_dir):
        path = os.path.join(args.exp_dir, fname)
        if os.path.isdir(path):
            for exp_name in set_exps:
                if str(exp_name) + "-" in fname:
                    assert os.listdir(path) == ["version_0"] or os.listdir(
                        path) == ["version_0", ".DS_Store"]
                    exp_paths.append(os.path.join(path, "version_0"))

    image_paths = []
    if args.inputs_and_targets:
        # Generate list of image paths for inputs
        inputs_fol = os.path.join(args.exp_dir, "inputs")
        for fname in sorted(os.listdir(inputs_fol)):
            if fname == ".DS_Store":
                continue
            if int(fname) not in args.imgs:
                continue
            path = os.path.join(inputs_fol, fname)
            if os.path.isdir(path):
                max_num = max(int(x[:6]) for x in os.listdir(path) if x != ".DS_Store")
                image_paths.append(os.path.join(path, format(max_num, '06d') + ".png"))

        # Generate list of image paths for targets
        targets_fol = os.path.join(args.exp_dir, "targets")
        if args.imgs:
            image_paths.extend([os.path.join(targets_fol, f)
                                for f in [sorted(os.listdir(targets_fol))[i] for i in args.imgs] if f.endswith('.png')])
        else:
            image_paths.extend([os.path.join(targets_fol, f)
                                for f in sorted(os.listdir(targets_fol)) if f.endswith('.png')])

    # Generate list of images in the experiment paths (use the last epoch)
    for top_exp_folder in exp_paths:
        dirs = [fname for fname in os.listdir(top_exp_folder) if (
            os.path.isdir(os.path.join(top_exp_folder, fname)) and "epoch" in fname)]

        max_epoch = max(int(x[6:]) for x in dirs)
        max_epoch_folder = "epoch_" + str(max_epoch)

        if args.imgs:
            image_paths.extend([os.path.join(top_exp_folder, max_epoch_folder, f)
                                for f in [sorted(os.listdir(os.path.join(top_exp_folder, max_epoch_folder)))[i] for i in args.imgs] if f.endswith('.png')])
        else:
            image_paths.extend([os.path.join(top_exp_folder, max_epoch_folder, f)
                                for f in sorted(os.listdir(os.path.join(top_exp_folder, max_epoch_folder))) if f.endswith('.png')])

    # Create and save image grid
    img = concat_images(image_paths, (IMAGE_WIDTH, IMAGE_HEIGHT),
                        (14+1, len(exp_paths)), exp_paths, args.padding, args.inputs_and_targets, args.imgs)

    img.show()
    img.save('_vs_'.join(set_exps)+'.png')
