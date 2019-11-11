import argparse
import os
import time
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf
import yaml

import utils


class Generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for feature, label in zip(hf["features"], hf["label"]):
                yield feature, label


def read_hdf(hdf5_path: str) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_generator(
        Generator(hdf5_path),
        (np.uint8, np.uint8),
        (tf.TensorShape([400, 951, 1]), tf.TensorShape([400, 951])))

    dataset = ds.make_one_shot_iterator()

    return dataset


def create_seeds(label: tf.Tensor,
                 num_classes: int,
                 size: int,
                 column: int):
    last_seed = 0
    horizon_depths = [None] * (num_classes - 1)
    for idx, seed_class in enumerate(range(num_classes - 1)):
        depth = np.argmax(label[:, column][last_seed:] > 0) + 1 + last_seed
        last_seed = depth + 2

        vertical_coordinate = depth - int(size / 2)
        horizontal_coordinate = column - int(size / 2)

        # protecting for vertical axis out of bound
        if vertical_coordinate < 0:
            vertical_coordinate = 0
        elif vertical_coordinate + size > label.shape[0]:
            vertical_coordinate = label.shape[0] - size

        # protecting for vertical axis out of bound
        if horizontal_coordinate < 0:
            horizontal_coordinate = 0
        elif horizontal_coordinate + size > label.shape[1]:
            horizontal_coordinate = label.shape[1] - size

        seed = (vertical_coordinate, horizontal_coordinate)
        horizon_depths[idx] = seed

    return horizon_depths


def line2tiles(feat: tf.Tensor,
               label: tf.Tensor,
               size: int,
               stride: int,
               num_classes: int):
    init_col = 150
    number_tiles = get_number_tiles(feat.shape[1], init_col, size, stride)

    model_input_list = [None] * number_tiles * (num_classes - 1)
    mini_label_list = [None] * number_tiles * (num_classes - 1)

    top_left_positions = [None] * number_tiles

    las_img_begin = feat.shape[1] - int(size / 2)
    for i, c in enumerate(range(init_col, las_img_begin, stride)):
        top_left_positions[i] = create_seeds(label, num_classes, size, column=c)

    for idx in range(len(top_left_positions)):
        for idy, tl_seed in enumerate(top_left_positions[idx]):
            mini_label_tile = label[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size]
            mini_feat_tile = feat[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size]

            if mini_label_tile.shape != (size, size):
                break

            center_class = mini_label_tile[int(size / 2), int(size / 2)].numpy()
            one_label_tile = mini_label_tile.numpy()
            one_label_tile[one_label_tile != center_class] = one_label_tile[mini_label_tile != center_class] * 0

            one_label_tile = utils.clean_label(tf.convert_to_tensor(one_label_tile, tf.uint8))

            mini_input_tile = tf.Variable(one_label_tile)

            if idx == len(top_left_positions) - 1:
                vertical_diff = 0
            else:
                vertical_diff = top_left_positions[idx][idy][0] - top_left_positions[idx + 1][idy][0]

            if np.abs(vertical_diff) > size:
                vertical_diff = 0

            if vertical_diff > 0:
                # horizontal zeros
                mini_input_tile = mini_input_tile[:vertical_diff, :].assign(tf.zeros((np.abs(vertical_diff), size),
                                                                                     tf.uint8))
            elif vertical_diff < 0:
                # horizontal zeros
                mini_input_tile = mini_input_tile[vertical_diff:, :].assign(tf.zeros((np.abs(vertical_diff), size),
                                                                                     tf.uint8))

            # vertical zeros
            mini_input_tile = mini_input_tile[:, -stride:].assign(tf.zeros((size, stride), tf.uint8))

            model_input = tf.stack([mini_feat_tile, mini_input_tile], axis=2)

            model_input_list[idx * (num_classes - 1) + idy] = model_input
            mini_label_list[idx * (num_classes - 1) + idy] = one_label_tile
    return model_input_list, mini_label_list


def get_number_tiles(image_size: int,
                     init_col: int,
                     tile_size: int,
                     stride: int):
    return int(((image_size - (init_col - int(tile_size / 2))) - tile_size) / stride) + 1


def prepare_sliding_window_binary_data(dataset: tf.data.Dataset,
                                       break_tiles_info: Tuple[int, int, int],
                                       num_records: int,
                                       num_classes: int,
                                       num_horizons: int = 8):
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    num_tiles = get_number_tiles(951, 150, size, stride)

    batch_input = [None] * num_tiles * num_records * (num_horizons - 1)
    batch_label = [None] * num_tiles * num_records * (num_horizons - 1)

    idx = 0
    count = 0
    print("Generating dataset...")
    for feat, label in dataset:
        tf_feat_0 = tf.reshape(feat, [feat.shape[0], feat.shape[1]])
        tf_label_0 = tf.reshape(label, [feat.shape[0], feat.shape[1]])

        count += 1
        print(f"\tProcessing inline: [{count}/{num_records}]")

        model_input_list, mini_label_list = line2tiles(tf_feat_0, tf_label_0, size, stride,
                                                       num_horizons)
        np.stack(model_input_list)
        np.stack(mini_label_list)

        batch_input[idx:(idx + len(model_input_list))] = model_input_list
        batch_label[idx:(idx + len(model_input_list))] = mini_label_list
        idx += len(model_input_list)

    model_input_batch = tf.stack(batch_input)
    mini_label_batch = tf.stack(batch_label)
    mini_label_batch = tf.keras.utils.to_categorical(mini_label_batch, num_classes=num_classes)

    return model_input_batch, mini_label_batch


def save_hdf(output_path: str,
             input_tiles: tf.Tensor,
             label_tiles: tf.Tensor):
    h5f = h5py.File(output_path, 'w')
    h5f.create_dataset('features', data=input_tiles, chunks=True)
    h5f.create_dataset('labels', data=label_tiles, chunks=True)
    h5f.close()


def create_dataset(dataset_path: str,
                   output_path: str):
    start = time.time()
    train_set = read_hdf(os.path.join(dataset_path, 'train.h5'))
    test_set = read_hdf(os.path.join(dataset_path, 'test.h5'))
    ds_info = yaml.safe_load(open(os.path.join(dataset_path, 'dataset_log.txt')))

    input_shape = (64, 64, 2)
    num_classes = 2

    seed = 0
    size = input_shape[0]
    stride = int(input_shape[0] / 8)

    break_tiles_info = (seed, size, stride)

    train_tiles_input, train_tiles_label = prepare_sliding_window_binary_data(train_set,
                                                                              break_tiles_info,
                                                                              ds_info['train_records'],
                                                                              num_classes)
    test_tiles_input, test_tiles_label = prepare_sliding_window_binary_data(test_set,
                                                                            break_tiles_info,
                                                                            ds_info['test_records'],
                                                                            num_classes)

    utils.makedir(output_path)
    save_hdf(os.path.join(output_path, 'train.h5'), train_tiles_input, train_tiles_label)
    save_hdf(os.path.join(output_path, 'test.h5'), test_tiles_input, test_tiles_label)

    print(f"Total time for dataset generation: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Path to the train HDF5 file.')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output train and test HDF5 dataset files.')

    args = parser.parse_args()
    create_dataset(args.dataset_path, args.output_dir)
