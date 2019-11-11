import os
from typing import Union

import tensorflow as tf


def makedir(path: str):
    """ Wrapper to create all the necessary directories in *path* with user/group rights to
        read, write and execute.
    Args:
        path: (str) path to the directory to be created.
    """
    os.makedirs(path, mode=0o770, exist_ok=True)


def clean_label(label_tile: Union[tf.Tensor, tf.Variable]) -> tf.Tensor:
    label_tile = label_tile.numpy()
    non_zeros = label_tile > 0
    label_tile[non_zeros] = 1

    label_tile = tf.convert_to_tensor(label_tile, tf.uint8)

    return label_tile
