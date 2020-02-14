import time

import ssd_mobilenet
from utils import draw_boxes
from utils import eprint
from utils import get_cpuinfo
from utils import load_jpg
from utils import preprocess

ssd_mobilenet.init()
detector = ssd_mobilenet.detector


def obj_detect(image_path):
    """return bounding box info."""
    img = load_jpg(image_path)
    image = preprocess(img)
    st_time = time.time()
    result = detector(image)
    end_time = time.time()
    eprint(
        "inference time for {} : {:.3f} seconds".format(image_path, end_time - st_time)
    )
    result = {key: value.numpy() for key, value in result.items()}
    return img, result


def boxed_img(img_path):
    img, result = obj_detect(img_path)
    return draw_boxes(
        img.numpy(),
        boxes=result["detection_boxes"],
        classes=result["detection_class_entities"],
        scores=result["detection_scores"],
    )


if __name__ == "__main__":
    img, yhat = obj_detect(image_path="./imgs/obj1.jpg")
    print(yhat)
