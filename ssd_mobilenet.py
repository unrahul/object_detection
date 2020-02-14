# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1.keras.backend import set_session

from utils import get_cpuinfo

num_cores, num_sockets = get_cpuinfo()
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["OMP_NUM_THREADS"] = str(num_cores)

set_session(
    Session(
        config=ConfigProto(
            intra_op_parallelism_threads=num_cores,
            inter_op_parallelism_threads=num_sockets,
        )
    )
)


def init():
    global detector
    model = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    model_path = "./models/"
    detector = hub.load(model_path).signatures["default"]
    detector(tf.zeros([1, 256, 256, 3], dtype=tf.dtypes.float32, name="init"))
    print("Loaded detector!")
