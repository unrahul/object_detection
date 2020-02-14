import glob
import os

from main import boxed_img, detector
from utils import display_image, draw_boxes, load_jpg, preprocess

os.environ["PYTHONPATH"] = "/workspace/"

for img_path in glob.glob("imgs/*.jpg"):
    bb_img = boxed_img(img_path)
    display_image(bb_img)
