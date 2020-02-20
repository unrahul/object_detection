import io
import time
from binascii import b2a_base64

from PIL import Image

import ssd_mobilenet
from utils import (draw_boxes, eprint, get_cpuinfo, img_buffer, load_jpg,
                   preprocess)

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


def boxed_img(img_data):
    """return base64 encoded boxed image."""
    if isinstance(img_data, str):
        img_path = img_data
    else:
        img_path = img_buffer(img_data)
    img, result = obj_detect(img_path)
    boxed_np_image = draw_boxes(
        img.numpy(),
        boxes=result["detection_boxes"],
        classes=result["detection_class_entities"],
        scores=result["detection_scores"],
    )
    result = Image.fromarray(boxed_np_image, "RGB")
    binary_buffer = io.BytesIO()
    result.save(binary_buffer, format="JPEG")
    return b2a_base64(binary_buffer.getvalue())


if __name__ == "__main__":
    img, yhat = obj_detect(image_path="./imgs/obj1.jpg")
    print(yhat)
