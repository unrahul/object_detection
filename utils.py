import sys
import subprocess
import psutil

import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import ImageColor, ImageDraw, ImageFont
import tensorflow as tf


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def get_cpuinfo():
    """return num of physical cores and sockets"""
    num_sockets = int(
        subprocess.check_output(
            "cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l", shell=True
        )
    )
    return psutil.cpu_count(logical=False), num_sockets


def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)


def load_jpg(path):
    """decode and return pixel data from jpeg file."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def preprocess(img):
    """expand dimension."""
    image = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return image


def draw_boxes(image, boxes, classes, scores):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    if boxes.any():
        eprint("object identified::")
    for i in range(min(boxes.shape[0], 10)):
        if scores[i] >= 0.10:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            string = "{}: {}%".format(classes[i].decode("ascii"), int(100 * scores[i]))
            eprint(string)
            color = colors[hash(classes[i]) % len(colors)]
            img = PIL.Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bbox(
                img, ymin, xmin, ymax, xmax, color, str_list=[string],
            )
            np.copyto(image, np.array(img))
    return image


def draw_bbox(img, ymin, xmin, ymax, xmax, color, str_list=()):
    """draw bounding box over an image."""

    font = ImageFont.truetype("/workspace/fonts/JosefinSans-SemiBold.ttf", 25)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    left, right = xmin * width, xmax * width
    top, bottom = ymin * height, ymax * height

    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=5,
        fill=color,
    )
    for string in str_list[::-1]:
        draw.text((left, top), string, fill="black", font=font)
