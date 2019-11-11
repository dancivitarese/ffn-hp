import time
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import exposure as exp

from models.model_zoo import unet_tf2


class Generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for feature, label in zip(hf["features"], hf["label"]):
                yield feature, label


def read_hdf(hdf5_path):
    ds = tf.data.Dataset.from_generator(
        Generator(hdf5_path),
        (np.uint8, np.uint8),
        (tf.TensorShape([400, 951, 1]), tf.TensorShape([400, 951])))

    dataset = ds.make_one_shot_iterator()

    return dataset


def read_tile_online(feat_tile, label_tile, seed, size, stride, model):
    v_seed = seed
    v_size = size
    v_stride = stride

    stride_label = stride

    data_flow = []

    for i in range(feat_tile.shape[0]):
        h_seed = seed
        h_size = size
        h_stride = stride

        mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if mini_label_tile.shape != (size, size):
            break

        for j in range(feat_tile.shape[1]):
            mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if mini_label_tile.shape != (size, size):
                break

            mini_input_tile = tf.Variable(mini_label_tile)
            mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))

            mini_feat_tile = feat_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

            mini_feat_reshape = tf.reshape(mini_feat_tile, [mini_feat_tile.shape[0], mini_feat_tile.shape[1], 1])
            mini_input_reshape = tf.reshape(mini_input_tile, [mini_feat_tile.shape[0], mini_feat_tile.shape[1], 1])
            model_input = tf.stack([mini_feat_reshape, mini_input_reshape], axis=2)
            model_input = tf.reshape(model_input, [1, mini_feat_tile.shape[0], mini_feat_tile.shape[1], 2])

            mini_label_tile_reshape = tf.reshape(mini_label_tile, [1, mini_feat_tile.shape[0], mini_feat_tile.shape[1]])

            model.fit(model_input, mini_label_tile_reshape, batch_size=1, epochs=1)

            # score = model.evaluate(model_input, mini_label_tile_reshape, verbose=0)

            model_input = model_input.numpy().astype('float32')

            result = model.predict(model_input, batch_size=1)

            mini_feat_tile_save = mini_feat_tile.numpy()
            mini_input_tile_save = mini_input_tile.numpy()
            mini_label_tile_save = mini_label_tile.numpy()

            result_reshape = tf.reshape(result, [size, size, 2])
            result_reshape = tf.unstack(result_reshape, axis=2)
            output_0 = result_reshape[0].numpy()
            output_1 = result_reshape[1].numpy()

            dict = {"mini_feat_tile_save": mini_feat_tile_save.tolist(),
                    "mini_input_tile_save": mini_input_tile_save.tolist(),
                    "mini_label_tile_save": mini_label_tile_save.tolist(), "output_0": output_0.tolist(),
                    "output_1": output_1.tolist()}

            data_flow.append(dict)

            h_seed += h_stride
        v_seed += v_stride

    return model, data_flow


def read_tile(feat_tile, label_tile, seed, size, stride):
    v_seed = seed
    v_size = size
    v_stride = stride

    stride_label = stride

    model_input_list = []
    mini_label_list = []

    for i in range(feat_tile.shape[0]):
        h_seed = seed
        h_size = size
        h_stride = stride

        mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if mini_label_tile.shape != (size, size):
            break

        for j in range(feat_tile.shape[1]):
            mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if mini_label_tile.shape != (size, size):
                break

            mini_input_tile = tf.Variable(mini_label_tile)
            mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))

            mini_feat_tile = feat_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

            model_input = tf.stack([mini_feat_tile, mini_input_tile], axis=2)

            model_input_list.append(model_input.numpy().astype('float32'))
            # mini_label_tile = tf.reshape(mini_label_tile, [mini_label_tile.shape[0], mini_label_tile.shape[1], 1])
            mini_label_list.append(mini_label_tile.numpy().astype('float32'))

            h_seed += h_stride
        v_seed += v_stride

    model_input_batch = tf.stack(model_input_list)
    mini_label_batch = tf.stack(mini_label_list)

    batch = {"model_input_batch": model_input_batch, "mini_label_batch": mini_label_batch}

    return batch


def break_tiles(feat_tile, label_tile, seed, size, stride):
    v_seed = seed
    v_size = size
    v_stride = stride

    stride_label = stride

    model_input_list = []
    mini_label_list = []

    for i in range(feat_tile.shape[0]):
        h_seed = seed
        h_size = size
        h_stride = stride

        mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if mini_label_tile.shape != (size, size):
            break

        for j in range(feat_tile.shape[1]):
            mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if mini_label_tile.shape != (size, size):
                break

            mini_input_tile = tf.Variable(mini_label_tile)
            mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))

            mini_feat_tile = feat_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

            model_input = tf.stack([mini_feat_tile, mini_input_tile], axis=2)

            model_input_list.append(model_input.numpy().astype('float32'))
            # mini_label_tile = tf.reshape(mini_label_tile, [mini_label_tile.shape[0], mini_label_tile.shape[1], 1])
            mini_label_list.append(mini_label_tile.numpy().astype('float32'))

            h_seed += h_stride
        v_seed += v_stride

    return model_input_list, mini_label_list


def break_feat_tile(feat_tile, seed, size, stride):
    v_seed = seed
    v_size = size
    v_stride = stride

    model_input_list = []

    for i in range(feat_tile.shape[0]):
        h_seed = seed
        h_size = size
        h_stride = stride

        mini_feat_tile = feat_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if mini_feat_tile.shape != (size, size):
            break

        for j in range(feat_tile.shape[1]):
            mini_feat_tile = feat_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if mini_feat_tile.shape != (size, size):
                break

            model_input_list.append(mini_feat_tile.numpy().astype('float32'))

            h_seed += h_stride
        v_seed += v_stride

    return model_input_list


def break_tiles_all_dir(feat_tile, label_tile, seed, size, stride):
    v_seed = seed
    v_size = size
    v_stride = stride

    stride_label = stride

    model_input_list = []
    mini_label_list = []

    directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']

    for i in range(feat_tile.shape[0]):
        h_seed = seed
        h_size = size
        h_stride = stride

        mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

        if mini_label_tile.shape != (size, size):
            break

        for j in range(feat_tile.shape[1]):
            mini_label_tile = label_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if mini_label_tile.shape != (size, size):
                break

            mini_input_tile = tf.Variable(mini_label_tile)

            direction = directions[int(np.random.random() * 100 % len(directions))]
            if direction == 'RIGHT':
                mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))
            elif direction == 'LEFT':
                mini_input_tile = mini_input_tile[:, :stride_label].assign(tf.zeros((size, stride), tf.uint8))
            elif direction == 'UP':
                mini_input_tile = mini_input_tile[:stride_label, :].assign(tf.zeros((stride, size), tf.uint8))
            elif direction == 'DOWN':
                mini_input_tile = mini_input_tile[- stride_label:, :].assign(tf.zeros((stride, size), tf.uint8))

            # mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))

            mini_feat_tile = feat_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

            model_input = tf.stack([mini_feat_tile, mini_input_tile], axis=2)

            model_input_list.append(model_input.numpy().astype('float32'))
            # mini_label_tile = tf.reshape(mini_label_tile, [mini_label_tile.shape[0], mini_label_tile.shape[1], 1])
            mini_label_list.append(mini_label_tile.numpy().astype('float32'))

            h_seed += h_stride
        v_seed += v_stride

    return model_input_list, mini_label_list


def break_sliding_tiles(feat, label, seed, size, stride, num_classes):
    v_seed = seed
    v_size = size
    v_stride = stride

    stride_label = stride

    model_input_list = []
    mini_label_list = []

    horizon = 1
    last_seed = 0

    # directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
    directions = ['RIGHT', 'UP', 'DOWN']

    for seed_class in range(num_classes):
        if seed_class == 0:
            continue

        for p in range(label.shape[0]):
            if p > last_seed:
                el = label[p, 0]
                if el == horizon:
                    break
        last_seed = p + 3

        v_seed = p - int(size / 2)

        h_seed = 0
        h_size = size
        h_stride = stride

        mini_label_tile = label[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

        if mini_label_tile.shape != (size, size):
            break

        for j in range(feat.shape[1]):
            mini_label_tile = label[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if mini_label_tile.shape != (size, size):
                break

            mini_input_tile = tf.Variable(mini_label_tile)

            direction = directions[int(np.random.random() * 100 % len(directions))]
            if direction == 'RIGHT':
                mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))
            # elif direction == 'LEFT':
            #     mini_input_tile = mini_input_tile[:, :stride_label].assign(tf.zeros((size, stride), tf.uint8))
            elif direction == 'UP':
                mini_input_tile = mini_input_tile[:stride_label, :].assign(tf.zeros((stride, size), tf.uint8))
            elif direction == 'DOWN':
                mini_input_tile = mini_input_tile[- stride_label:, :].assign(tf.zeros((stride, size), tf.uint8))

            # mini_input_tile = mini_input_tile[:, - stride_label:].assign(tf.zeros((size, stride), tf.uint8))

            mini_feat_tile = feat[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

            model_input = tf.stack([mini_feat_tile, mini_input_tile], axis=2)

            model_input_list.append(model_input.numpy().astype('float32'))
            # mini_label_tile = tf.reshape(mini_label_tile, [mini_label_tile.shape[0], mini_label_tile.shape[1], 1])
            mini_label_list.append(mini_label_tile.numpy().astype('float32'))

            for k in range(size):
                el = mini_label_tile[k, -1]
                if el == horizon:
                    break

            if k == (size - 1):
                h_seed += h_stride
                continue

            new_seed = k
            if new_seed > size / 2:
                v_seed += stride
            elif new_seed < size / 2:
                v_seed -= stride
            # v_seed = v_seed + new_seed - v_stride
            if v_seed < 0:
                v_seed = 0
            elif v_seed + size > feat.shape[0]:
                v_seed = feat.shape[0] - size

            if (new_seed < (size / 2) + stride) and (new_seed > (size / 2) - stride):
                h_seed += h_stride

    return model_input_list, mini_label_list


def line2tiles(feat, label, seed, size, stride, num_classes):
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

            one_label_tile = clean_label(tf.convert_to_tensor(one_label_tile, tf.uint8))

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


def get_number_tiles(image_size, init_col, tile_size, stride):
    return int(((image_size - (init_col - int(tile_size / 2))) - tile_size) / stride) + 1


def create_seeds(label, num_classes, size, column):
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


def get_dataset(images_dataset, break_tiles_info, num_classes):
    # tiles_dataset = prepare_data(images_dataset, break_tiles_info, num_classes)
    # tiles_dataset = prepare_sliding_window_data(images_dataset, break_tiles_info, num_classes)
    tiles_dataset = prepare_sliding_window_binary_data(images_dataset, break_tiles_info, num_classes)
    dataset = tf.data.Dataset.from_tensor_slices(tiles_dataset)

    return dataset


def prepare_data(dataset, break_tiles_info, num_classes):
    batch_input = []
    batch_label = []

    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    for feat, label in dataset:
        tf_feat_0 = tf.reshape(feat, [400, 951])
        tf_label_0 = tf.reshape(label, [400, 951])

        # model_input_list, mini_label_list = break_tiles(tf_feat_0, tf_label_0, seed, size, stride)
        model_input_list, mini_label_list = break_tiles_all_dir(tf_feat_0, tf_label_0, seed, size, stride)

        batch_input += model_input_list
        batch_label += mini_label_list

    model_input_batch = tf.stack(batch_input)
    mini_label_batch = tf.stack(batch_label)
    batch = {"model_input_batch": model_input_batch, "mini_label_batch": mini_label_batch}

    batch["mini_label_batch"] = tf.keras.utils.to_categorical(batch["mini_label_batch"], num_classes=num_classes)

    return batch["model_input_batch"], batch["mini_label_batch"]


def prepare_sliding_window_data(dataset, break_tiles_info, num_classes):
    batch_input = []
    batch_label = []

    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    for feat, label in dataset:
        tf_feat_0 = tf.reshape(feat, [400, 951])
        tf_label_0 = tf.reshape(label, [400, 951])

        model_input_list, mini_label_list = break_sliding_tiles(tf_feat_0, tf_label_0, seed, size, stride, num_classes)

        batch_input += model_input_list
        batch_label += mini_label_list

    model_input_batch = tf.stack(batch_input)
    mini_label_batch = tf.stack(batch_label)
    batch = {"model_input_batch": model_input_batch, "mini_label_batch": mini_label_batch}

    batch["mini_label_batch"] = tf.keras.utils.to_categorical(batch["mini_label_batch"], num_classes=num_classes)

    return batch["model_input_batch"], batch["mini_label_batch"]


def prepare_sliding_window_binary_data(dt, dataset, break_tiles_info, num_classes, num_horizons=8):
    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    num_tiles = get_number_tiles(951, 150, size, stride)

    # dataset_size = 59 #len(list(dataset))

    dataset_size = 0
    for i, l in dt:
        dataset_size += 1

    batch_input = [None] * num_tiles * dataset_size * (num_horizons - 1)
    batch_label = [None] * num_tiles * dataset_size * (num_horizons - 1)

    idx = 0
    count = 0
    print("Generating dataset...")
    for feat, label in dataset:
        tf_feat_0 = tf.reshape(feat, [400, 951])
        tf_label_0 = tf.reshape(label, [400, 951])

        # tf_label_0 = clean_label(tf_label_0)

        count += 1
        print(f"\tProcessing inline: [{count}/{dataset_size}]")

        model_input_list, mini_label_list = line2tiles(tf_feat_0, tf_label_0, seed, size, stride,
                                                       num_horizons)
        np.stack(model_input_list)
        np.stack(mini_label_list)

        batch_input[idx:(idx + len(model_input_list))] = model_input_list
        batch_label[idx:(idx + len(model_input_list))] = mini_label_list
        idx += len(model_input_list)

    model_input_batch = tf.stack(batch_input)
    mini_label_batch = tf.stack(batch_label)
    batch = {"model_input_batch": model_input_batch, "mini_label_batch": mini_label_batch}

    batch["mini_label_batch"] = tf.keras.utils.to_categorical(batch["mini_label_batch"], num_classes=num_classes)

    return batch["model_input_batch"], batch["mini_label_batch"]


def save_model(model, model_name):
    model.save(model_name)
    print("Saved model to disk")


def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    print("Loaded model from disk")

    return model


def evaluate_model(model, x, y):
    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = model.evaluate(x, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def clean_label(label_tile):
    label_tile = label_tile.numpy()
    non_zeros = label_tile > 0
    label_tile[non_zeros] = 1

    label_tile = tf.convert_to_tensor(label_tile, tf.uint8)

    return label_tile


def online_trainning():
    hdf5_path = "/Users/gabriel/Documents/Dataset/dataset-10/train.h5"
    dataset = read_hdf(hdf5_path)

    model = danet2(output_channels=2)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

    for feat, label in dataset:
        tf_feat_0 = tf.reshape(feat, [400, 951])
        tf_label_0 = tf.reshape(label, [400, 951])

        tf_label_0 = clean_label(tf_label_0)

        # sample_image, sample_mask = feat, label
        # display([sample_image, sample_mask])

        seed = 0
        size = 40
        stride = 25

        model, data_flow = read_tile_online(tf_feat_0, tf_label_0, seed, size, stride, model)

        save_model(model)


def get_prediction_tiles(predictions):
    return np.argmax(predictions, axis=3)


def _get_num_tiles(shape: Tuple[int, ...], window_shape: Tuple[int, ...],
                   stride: Tuple[int, ...]) -> (Tuple[int, ...], Tuple[int, ...]):
    """ Gets the number of tiles that fits in *length* points and the remaining points to the
        end of the *length* value.

    Args:
        shape: original array shape.
        window_shape: tuple containing the shape of the sliding window.
        stride: tuple containing the step sizes for the sliding window.

    Returns:
        list containing the number of tiles.
        list containing the number of remaining points to the end of the length value.
    """

    def _num_tiles(length: int, window_size: int, stride: int) -> Tuple[int, int]:
        """ Gets the number of tiles that fits in *length* points and the remaining points to the
            end of the *length* value.

        Args:
            length: original array size.
            window_size: (int) size of the window.
            stride: (int) step size of the sliding window.

        Returns:
            (int32) number of tiles.
            (int32) number of remaining points to the end of the length value.
        """

        total_tiles = ((length - window_size) // stride) + 1
        remainder_points = (length - window_size) % stride

        return total_tiles, remainder_points

    assert len(stride) == len(window_shape) == len(shape), \
        "*strides*, and *window_shape* and *shape* must have the same size!"

    num_tiles = [0] * len(stride)
    remain_points = [0] * len(stride)

    for i in range(len(num_tiles)):
        num_tiles[i], remain_points[i] = _num_tiles(shape[i], window_shape[i], stride[i])

    return tuple(num_tiles), tuple(remain_points)


def reconstruct_from_tiles_2d(tile_array: np.ndarray, final_shape: Tuple[int, int],
                              stride: Tuple[int, int]) -> np.ndarray:
    """ Reconstruct an 2D array from the tiles in *tile_array* skipping *stride* in each dimension. Assumes that
        *tile_array* comes exactly from an image with shape *final_shape*. In other words, when breaking this image into
        the *tile_array* no pixels should be left over.

    Args:
        tile_array: ndarray containing the tiles.
        final_shape: original array shape.
        stride: tuple containing the number of points to skip in each dimension.

    Returns:
        reconstructed array
    """

    assert len(tile_array.shape) == 3

    tile_shape = tile_array.shape[1:]
    tiles_per_dim, _ = _get_num_tiles(shape=final_shape, window_shape=tile_shape, stride=stride)

    final_array = np.zeros(final_shape, dtype=tile_array.dtype)

    idx = 0
    rows = range(0, tiles_per_dim[0] * stride[0], stride[0])
    for row in rows:
        columns = range(0, tiles_per_dim[1] * stride[1], stride[1])
        for col in columns:
            final_array[row:row + tile_shape[0], col:col + tile_shape[1]] = tile_array[idx]
            idx += 1

    return final_array


def scale_intensity(image: np.ndarray, gray_levels: int, percentile: float) -> np.ndarray:
    """ Rescale *image* intensity to specified number of gray levels. Before rescaling,
        remove percentile outliers. Final result is in uint8 interval (0 - 255) with
        *gray_levels* different levels.

    Args:
        image: numpy array of floats.
        gray_levels: (int) number of levels of quantization.
        percentile: (float) percentile of outliers to be cut (from 0.0 to 100.0)

    Returns:
        image in uint8 format, quantized to *gray_levels* levels.
    """

    pmin, pmax = np.percentile(image, (percentile, 100.0 - percentile))
    im = exp.rescale_intensity(image, in_range=(pmin, pmax),
                               out_range=(0, gray_levels - 1))

    out = exp.rescale_intensity(im, in_range=(0, gray_levels - 1), out_range=(0, 255)).astype(np.uint8)

    return out


def run_model(feat, label, model, break_tiles_info):
    tf_feat_0 = tf.reshape(feat, [400, 951])
    tf_label_0 = tf.reshape(label, [400, 951])

    # sample_image, sample_mask = feat, label
    # display([sample_image, sample_mask])

    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    batch = read_tile(tf_feat_0, tf_label_0, seed, size, stride)

    predictions = model.predict(batch["model_input_batch"], batch_size=batch["model_input_batch"].shape[0])

    tile_predictions = get_prediction_tiles(predictions)

    return reconstruct_from_tiles_2d(tile_predictions, (400, 951), (stride, stride))


def running_through_pred(input_seed, model, dataset, break_tiles_info):
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    feat, _ = next(dataset)

    tf_feat_0 = tf.reshape(feat, [400, 951])

    model_input_list = break_feat_tile(tf_feat_0, *break_tiles_info)

    tile_predictions = running_loop(input_seed, model_input_list, model, size, stride)
    tile_predictions = np.stack(tile_predictions)

    output = reconstruct_from_tiles_2d(tile_predictions, (400, 951), (stride, stride))
    return output


def running_loop(seed, model_input_list, model, size, stride):
    tile_predictions = [None] * len(model_input_list)
    mask_tile = seed

    for idx, input_tile in enumerate(model_input_list):
        model_input = tf.stack([input_tile, mask_tile], axis=2)
        model_input = tf.reshape(model_input, [1, size, size, 2])
        prediction = model(model_input)
        output_tile = tf.reshape(get_prediction_tiles(prediction.numpy()), [size, size])
        tile_predictions[idx] = output_tile
        mask_tile = tf.Variable(output_tile)[:, - stride:].assign(tf.zeros((size, stride), tf.int64))
        mask_tile = tf.dtypes.cast(mask_tile, tf.uint8)

    return tile_predictions


def train_model(train_dataset, test_dataset, model, callbacks, epochs):
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)
    # model.fit(train_dataset[0], train_dataset[1], epochs=epochs, validation_data=(test_dataset[0], test_dataset[1]),
    #           callbacks=callbacks, batch_size=32)
    # model.fit(train_dataset[0], train_dataset[1], epochs=epochs, batch_size=32)  # , validation_data=test_dataset)
    return model


def training_loop(train_dataset, val_dataset, model, model_name, optimizer, loss_fn, metrics, epochs):
    train_metrics = metrics[0]
    val_metrics = metrics[1]

    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        epoch_loss = 0
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                pixel_loss = loss_fn(y_batch_train, logits)
                w = y_batch_train[:, :, :, -1] * 9
                w = w + y_batch_train[:, :, :, 0]
                pixel_loss *= w
                loss_value = tf.reduce_sum(pixel_loss) * (
                        1. / y_batch_train.shape[0] / y_batch_train.shape[1] / y_batch_train.shape[2])
                # y_batch_reshape = tf.reshape(y_batch_train, [y_batch_train.shape[0], y_batch_train.shape[1],
                #                                              y_batch_train.shape[2], 1])
                # pixel_loss = loss_fn(y_batch_reshape, logits)
                # w = y_batch_train.numpy() * 9
                # w[w == 0] = 1
                # loss = pixel_loss * w
                # loss_value = tf.reduce_sum(loss) * (
                #         1. / y_batch_train.shape[0] / y_batch_train.shape[1] / y_batch_train.shape[2])

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            epoch_loss += loss_value

            # train_acc_metric(y_batch_train, logits)
            # Update training metric.
            for metric in train_metrics.values():
                metric(y_batch_train, logits)
                # metric(y_batch_reshape, logits)

            # Log every 2 batches.
            if step % 2 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * x_batch_train.shape[0]))

        print(f"Epoch loss: {epoch_loss / (step + 1)}")
        # Display metrics at the end of each epoch.
        for metric_name, metric in train_metrics.items():
            print(f"| {metric_name}: {metric.result()} ", end="", flush=True)
            metric.reset_states()
        print("|")

        # train_acc = train_acc_metric.result()
        # print('Training acc over epoch: %s' % (float(train_acc),))
        # # Reset training metrics at the end of each epoch
        # train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            # val_loss_value = loss_fn(y_batch_val, val_logits)
            # Update val metrics
            # y_batch_val_reshape = tf.reshape(y_batch_val, [y_batch_val.shape[0], y_batch_val.shape[1],
            #                                                y_batch_val.shape[2], 1])
            for metric in val_metrics.values():
                metric(y_batch_val, val_logits)
                # metric(y_batch_val_reshape, val_logits)

        for metric_name, metric in val_metrics.items():
            print(f"| {metric_name}: {metric.result()} ", end="", flush=True)
            metric.reset_states()
        print("|")
        #     val_acc_metric(y_batch_val, val_logits)
        # val_acc = val_acc_metric.result()
        # val_acc_metric.reset_states()
        # print('Validation acc: %s' % (float(val_acc),))

        model.save(f'{model_name}_{epoch}.h5')
        is_best = False
        if is_best:
            model.save(model_name)

    return model


def iou(y_true, y_pred):
    m = tf.keras.metrics.MeanIoU(num_classes=8)
    m.update_state(tf.keras.backend.argmax(y_true, axis=-1), tf.keras.backend.argmax(y_pred, axis=-1))
    return f'IoU:  {m.result()}'


class MeanIoU(tf.keras.metrics.Metric):

    def __init__(self, name='mean_iou', **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.tf_mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)

    # def __call__(self, y_true, y_pred, sample_weight=None):
    #     self.update_state(y_true, y_pred, sample_weight=None)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        # y_true = y_true.numpy().round()
        # y_pred = y_pred.numpy().round()

        self.tf_mean_iou.update_state(y_true, y_pred)

    def result(self):
        return self.tf_mean_iou.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.tf_mean_iou.reset_states()


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a = tf.convert_to_tensor(a.numpy()[p])
    b = tf.convert_to_tensor(b[p])
    return a, b


def main():
    # hdf5_train_path = "/Users/gabriel/Documents/Dataset/dataset-10-fill/train.h5"
    # train_set = read_hdf(hdf5_train_path)
    #
    # hdf5_test_path = "/Users/gabriel/Documents/Dataset/dataset-10-fill/test.h5"
    # test_set = read_hdf(hdf5_test_path)
    #
    input_shape = (80, 80, 2)
    # model = unet_tf2(input_shape=input_shape, output_channels=8)
    #
    # # tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['categorical_accuracy'])
    #
    seed = 0
    size = input_shape[0]
    stride = int(input_shape[0] / 2 + 5)

    break_tiles_info = (seed, size, stride)
    #
    # train_dataset = get_dataset(train_set, break_tiles_info, num_classes=8, batch_size=32)
    # # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # test_dataset = get_dataset(test_set, break_tiles_info, num_classes=8, batch_size=32)
    # # train_dataset = prepare_data(train_set, break_tiles_info, num_classes=8)
    # # test_dataset = prepare_data(test_set, break_tiles_info, num_classes=8)
    # # train_dataset = unison_shuffled_copies(*train_dataset)
    #
    # LR_SCHEDULE = [
    #     # (epoch to start, learning rate) tuples
    #     (4, 0.0001)  # , (6, 0.01), (9, 0.005), (12, 0.001)
    # ]
    #
    # def lr_schedule(epoch, lr):
    #     """Helper function to retrieve the scheduled learning rate based on epoch."""
    #     if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
    #         return lr
    #     for i in range(len(LR_SCHEDULE)):
    #         if epoch == LR_SCHEDULE[i][0]:
    #             return LR_SCHEDULE[i][1]
    #     return lr
    #
    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath='a_new_model_{epoch}.h5',
    #         # Path where to save the model
    #         # The two parameters below mean that we will overwrite
    #         # the current checkpoint if and only if
    #         # the `val_loss` score has improved.
    #         save_best_only=True,
    #         monitor='val_loss',
    #         verbose=1)  # ,
    #     # tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # ]
    #
    # model = train_model(train_dataset, test_dataset, model, callbacks, epochs=5)

    # model = train_model(train_dataset, None, model, epochs=5)
    # save_model(model, "ffn_model_end.h5")
    # model = load_model("a_new_model_5.h5")
    model = load_model("test_model.h5")

    hdf5_valid_path = "/Users/gabriel/Documents/Dataset/dataset-10-fill/valid.h5"
    valid_dataset = read_hdf(hdf5_valid_path)
    # feat, label = next(valid_dataset)
    # output = run_model(feat, label, model, break_tiles_info)

    zero_seed = tf.zeros((size, size), tf.uint8)
    # model_seed = tf.Variable(zero_seed)[0][0] = 1

    output = running_through_pred(zero_seed, model, valid_dataset, break_tiles_info)

    feat, label = next(read_hdf(hdf5_valid_path))
    feat_img = Image.fromarray(tf.reshape(feat, [400, 951]).numpy())
    feat_img.save('feat_img.png')
    # label_img = Image.fromarray(label.numpy())
    # label_img.save('a_label_img.png')
    # output_img = Image.fromarray(output.astype(np.uint8))
    # output_img.save('a_output_image.png')

    result_O = scale_intensity(output.astype(np.uint8), 8, 5.0)
    img_O = Image.fromarray(result_O)
    img_O.save('pred_fill.png')
    result_L = scale_intensity(label.numpy(), 8, 5.0)
    img_L = Image.fromarray(result_L)
    img_L.save('label_fill.png')


def test_training_loop():
    hdf5_train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train.h5"
    train_set = read_hdf(hdf5_train_path)

    hdf5_test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test.h5"
    test_set = read_hdf(hdf5_test_path)

    input_shape = (16, 16, 2)
    num_classes_output = 1
    model = unet_tf2(input_shape=input_shape, output_channels=num_classes_output)
    num_classes = 8
    # model = load_model("trained_models/model_16w_s8_10ep_6.h5")

    # tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['categorical_accuracy'])

    seed = 0
    size = input_shape[0]
    stride = 2

    break_tiles_info = (seed, size, stride)

    # feat_tr, label_tr = next(train_set)
    # train_dataset = get_dataset([(feat_tr, label_tr)], break_tiles_info, num_classes=8)
    train_dataset = get_dataset(train_set, break_tiles_info, num_classes=num_classes)
    train_dataset = train_dataset.shuffle(buffer_size=10192).batch(batch_size=2048)
    # feat_ts, label_ts = next(test_set)
    # test_dataset = get_dataset([(feat_ts, label_ts)], break_tiles_info, num_classes=8)
    test_dataset = get_dataset(test_set, break_tiles_info, num_classes=num_classes)
    test_dataset = test_dataset.batch(batch_size=512)

    # train_dataset = prepare_data(train_set, break_tiles_info, num_classes=8)
    # test_dataset = prepare_data(test_set, break_tiles_info, num_classes=8)
    # train_dataset = unison_shuffled_copies(*train_dataset)

    # model = train_model(train_dataset, test_dataset, model, epochs=5)
    model_name = "trained_models/model_16w_s2_bin_bal"
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_metrics = {  # 'train_mean_iou': MeanIoU(),
        'train_cat_acc': tf.keras.metrics.BinaryAccuracy()}
    val_metrics = {  # 'val_mean_iou': MeanIoU(),
        'val_cat_acc': tf.keras.metrics.BinaryAccuracy()}
    metrics = (train_metrics, val_metrics)
    model = training_loop(train_dataset, test_dataset, model, model_name, optimizer, loss_fn, metrics, epochs=10)
    model.save(f'{model_name}.h5')
    # save_model(model, "test_model.h5")


def run_inside_sliding_window(seed_class, canvas, model, dataset, break_tiles_info):
    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    idx = seed

    feat, label = next(dataset)

    feat = tf.reshape(feat, [400, 951])

    v_seed = seed
    v_size = size
    v_stride = stride

    canvas_tile = tf.Variable(canvas)
    confidence = 0

    h_seed = 0
    h_size = size
    h_stride = stride
    for i in range(feat.shape[1]):

        feat_window = feat[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if feat_window.shape != (size, size):
            break

        canvas_window = canvas_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

        mini_feat_reshape = tf.reshape(feat_window, [feat_window.shape[0], feat_window.shape[1], 1])
        canvas_window_reshape = tf.reshape(canvas_window, [feat_window.shape[0], feat_window.shape[1], 1])
        model_input = tf.stack([mini_feat_reshape, canvas_window_reshape], axis=2)
        model_input = tf.reshape(model_input, [1, feat_window.shape[0], feat_window.shape[1], 2])

        prediction = model(model_input)
        output_tile = tf.reshape(get_prediction_tiles(prediction.numpy()), [size, size])

        output_tile = tf.dtypes.cast(output_tile, tf.uint8, name=None)
        canvas_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size].assign(output_tile)

        for j in range(size):
            el = output_tile[j, -1]
            if el == seed_class:
                confidence += 1
                if confidence == 2:
                    # idx = j
                    confidence = 0
                    break

        # new_seed = idx
        new_seed = j - 2
        if new_seed > size / 2:
            v_seed += stride
        elif new_seed < size / 2:
            v_seed -= stride

        # v_seed = v_seed + new_seed - v_stride
        if v_seed < 0:
            v_seed = 0
        elif v_seed + size > feat.shape[0]:
            v_seed = feat.shape[0] - size

        if (new_seed < (size / 2) + (2 * stride)) and (new_seed > (size / 2) - (2 * stride)):
            h_seed += h_stride
        # v_seed += v_stride

    return canvas_tile.numpy()


def run_centering_sliding_window_binary(canvas, model, dataset, break_tiles_info):
    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    feat, label = next(dataset)
    feat = tf.reshape(feat, [400, 951])

    canvas_tile = tf.Variable(canvas)

    v_seed = seed

    old_v_center = seed

    for h_seed in range(0, (feat.shape[1] - size), stride):
        tl_seed = (v_seed, h_seed)

        feat_window = feat[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size]
        canvas_window = canvas_tile[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size]

        if feat_window.shape != (size, size):
            break

        mini_feat_reshape = tf.reshape(feat_window, [feat_window.shape[0], feat_window.shape[1], 1])
        canvas_window_reshape = tf.reshape(canvas_window, [feat_window.shape[0], feat_window.shape[1], 1])
        model_input = tf.stack([mini_feat_reshape, canvas_window_reshape], axis=2)
        model_input = tf.reshape(model_input, [1, feat_window.shape[0], feat_window.shape[1], 2])

        prediction = model(model_input)
        output_tile = tf.reshape(get_prediction_tiles(prediction.numpy()), [size, size])

        if h_seed % 64 == 0:
            R = canvas_tile.numpy().astype(np.uint8) * 255
            G = label.numpy().astype(np.uint8) * 255
            B = np.zeros([feat.shape[0], feat.shape[1]], dtype=np.uint8)
            RGB = np.stack((R, G, B), axis=-1)
            Image.fromarray(RGB).show()

        output_tile = tf.dtypes.cast(output_tile, tf.uint8, name=None)
        canvas_tile[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size].assign(output_tile)

        next_v_center = np.argmax(canvas_tile[:, h_seed + int(size / 2) + stride] > 0) + 1
        if np.abs(old_v_center - next_v_center) <= size:
            v_seed = next_v_center - int(size / 2)
            old_v_center = next_v_center

        if v_seed < 0:
            v_seed = 0
        elif v_seed > feat.shape[0] + size:
            v_seed = feat.shape[0] - size

    return canvas_tile.numpy()


def run_centering_sliding_window_binary_using_label(canvas, model, dataset, break_tiles_info):
    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    feat, label = next(dataset)
    feat = tf.reshape(feat, [400, 951])
    label_cl = clean_label(label)

    canvas_tile = tf.Variable(canvas)

    v_seed = seed

    old_v_center = seed

    for h_seed in range(0, (feat.shape[1] - size), stride):
        tl_seed = (v_seed, h_seed)

        feat_window = feat[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size]
        canvas_window = canvas_tile[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size]

        if feat_window.shape != (size, size):
            break

        mini_feat_reshape = tf.reshape(feat_window, [feat_window.shape[0], feat_window.shape[1], 1])
        canvas_window_reshape = tf.reshape(canvas_window, [feat_window.shape[0], feat_window.shape[1], 1])
        model_input = tf.stack([mini_feat_reshape, canvas_window_reshape], axis=2)
        model_input = tf.reshape(model_input, [1, feat_window.shape[0], feat_window.shape[1], 2])

        prediction = model(model_input)
        output_tile = tf.reshape(get_prediction_tiles(prediction.numpy()), [size, size])

        output_tile = tf.dtypes.cast(output_tile, tf.uint8, name=None)
        canvas_tile[tl_seed[0]: tl_seed[0] + size, tl_seed[1]: tl_seed[1] + size].assign(output_tile)

        next_v_center = np.argmax(label_cl[:, h_seed + int(size / 2) + stride][(old_v_center - 2 * size):] > 0) + 1 + \
                        (old_v_center - 2 * size)
        if np.abs(old_v_center - next_v_center) <= size:
            v_seed = next_v_center - int(size / 2)
            old_v_center = next_v_center

        if v_seed < 0:
            v_seed = 0
        elif v_seed > feat.shape[0] + size:
            v_seed = feat.shape[0] - size

    return canvas_tile.numpy()


def run_through_canvas(canvas, model, dataset, break_tiles_info):
    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    feat, _ = next(dataset)

    feat = tf.reshape(feat, [400, 951])

    v_seed = seed
    v_size = size
    v_stride = stride

    canvas_tile = tf.Variable(canvas)

    for i in range(feat.shape[0]):
        h_seed = seed
        h_size = size
        h_stride = stride

        feat_window = feat[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if feat_window.shape != (size, size):
            break

        for j in range(feat.shape[1]):
            feat_window = feat[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
            if feat_window.shape != (size, size):
                break

            canvas_window = canvas_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size]

            mini_feat_reshape = tf.reshape(feat_window, [feat_window.shape[0], feat_window.shape[1], 1])
            canvas_window_reshape = tf.reshape(canvas_window, [feat_window.shape[0], feat_window.shape[1], 1])
            model_input = tf.stack([mini_feat_reshape, canvas_window_reshape], axis=2)
            model_input = tf.reshape(model_input, [1, feat_window.shape[0], feat_window.shape[1], 2])

            prediction = model(model_input)
            output_tile = tf.reshape(get_prediction_tiles(prediction.numpy()), [size, size])

            output_tile = tf.dtypes.cast(output_tile, tf.uint8, name=None)
            canvas_tile[v_seed: v_seed + v_size, h_seed: h_seed + h_size].assign(output_tile)

            h_seed += h_stride
        v_seed += v_stride

    return canvas_tile.numpy()


def get_input_seed(label_img, canvas, seed_class, size):
    input_seed = 0
    for i in range(label_img.shape[0]):
        el = label_img[i, 0]
        if el == seed_class:
            input_seed = i
            break
    seed_tile = label_img[input_seed - int(size / 2): input_seed + int(size / 2), 0: size]
    canvas_tile = tf.Variable(canvas)
    seed_tile = tf.dtypes.cast(seed_tile, tf.uint8, name=None)
    canvas_tile[input_seed, 0].assign(1)
    return input_seed, canvas_tile


def test_run_inside_sliding_window_with_seed_tile():
    input_shape = (64, 64, 2)

    size = input_shape[0]
    # stride = int(input_shape[0] / 8)
    stride = 16

    model = load_model("trained_models/model_64w_s16_bin_bal_centered_one-hrz-tiles.h5")

    hdf5_valid_path = "/Users/gabriel/Documents/Dataset/dataset-10/valid.h5"
    valid_dataset = read_hdf(hdf5_valid_path)

    # zero_seed = tf.zeros((size, size), tf.uint8)
    # model_seed = tf.Variable(zero_seed)[0][0] = 1

    _, label_img = next(read_hdf(hdf5_valid_path))
    canvas = tf.zeros((400, 951), tf.uint8)
    seed_class = 6

    input_seed, canvas = get_input_seed(label_img, canvas, seed_class, size)
    canvas = clean_label(canvas)

    seed = input_seed - int(size / 2)

    break_tiles_info = (seed, size, stride)

    # bin_seed_class = 1
    # output = run_inside_sliding_window(bin_seed_class, canvas, model, valid_dataset, break_tiles_info)
    output = run_centering_sliding_window_binary(canvas, model, valid_dataset, break_tiles_info)
    # output = run_centering_sliding_window_binary_using_label(canvas, model, valid_dataset, break_tiles_info)

    feat, label = next(read_hdf(hdf5_valid_path))
    feat_img = Image.fromarray(tf.reshape(feat, [400, 951]).numpy())
    feat_img.save('output_results/feat_img.png')
    # label_img = Image.fromarray(label.numpy())
    # label_img.save('a_label_img.png')
    # output_img = Image.fromarray(output.astype(np.uint8))
    # output_img.save('a_output_image.png')
    print("before scaling output")
    # result_O = scale_intensity(output.astype(np.uint8), 8, 5.0)
    # img_O = Image.fromarray(result_O)
    # img_O.save('output_results/s16w_h7_pred_fill.png')
    r0 = output.astype(np.uint8) * 175
    imgr0 = Image.fromarray(r0)
    imgr0.save('output_results/hrz6_64w_s16_seed_bin_centered_one-hrz-tiles_window_gpu.png')
    print("after scaling output")
    result_L = scale_intensity(label.numpy(), 8, 5.0)
    img_L = Image.fromarray(result_L)
    img_L.save('output_results/label_fill.png')


def run_slide_through_label(seed_class, label, break_tiles_info):
    output_frame = tf.zeros((400, 951), tf.uint8)
    output_horizon = tf.Variable(output_frame)

    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    idx = seed

    v_seed = seed
    v_size = size
    v_stride = stride

    confidence = 0

    h_seed = 0
    h_size = size
    h_stride = stride
    for i in range(label.shape[1]):

        label_window = label[v_seed: v_seed + v_size, h_seed: h_seed + h_size]
        if label_window.shape != (size, size):
            break

        output_horizon[v_seed: v_seed + v_size, h_seed: h_seed + h_size].assign(label_window)

        for j in range(size):
            el = label_window[j, -1]
            if el == seed_class:
                confidence += 1
                if confidence == 3:
                    # idx = j
                    confidence = 0
                    break

        # new_seed = idx
        new_seed = j - 3
        if new_seed > size / 2:
            v_seed += stride
        elif new_seed < size / 2:
            v_seed -= stride

        # v_seed = v_seed + new_seed - v_stride
        if v_seed < 0:
            v_seed = 0
        elif v_seed + size > label.shape[0]:
            v_seed = label.shape[0] - size

        if (new_seed < (size / 2) + stride) and (new_seed > (size / 2) - stride):
            h_seed += h_stride
        # v_seed += v_stride

    return output_horizon.numpy()


def test_sliding_through_label():
    input_shape = (16, 16, 2)

    size = input_shape[0]
    # stride = int(input_shape[0] / 10)
    stride = 2

    hdf5_valid_path = "/Users/gabriel/Documents/Dataset/dataset-10/valid.h5"
    valid_dataset = read_hdf(hdf5_valid_path)

    feat, label = next(valid_dataset)
    canvas = tf.zeros((400, 951), tf.uint8)
    seed_class = 1
    input_seed, canvas = get_input_seed(label, canvas, seed_class, size)

    seed = input_seed - int(size / 2)

    break_tiles_info = (seed, size, stride)
    output = run_slide_through_label(seed_class, label, break_tiles_info)
    feat_img = Image.fromarray(tf.reshape(feat, [400, 951]).numpy())
    feat_img.save('output_results/feat_img.png')
    # label_img = Image.fromarray(label.numpy())
    # label_img.save('a_label_img.png')
    # output_img = Image.fromarray(output.astype(np.uint8))
    # output_img.save('a_output_image.png')
    print("before scaling output")
    # result_O = scale_intensity(output.astype(np.uint8), 8, 5.0)
    # img_O = Image.fromarray(result_O)
    # img_O.save('output_results/hrz2_label.png')
    r0 = output.astype(np.uint8) * 125
    imgr0 = Image.fromarray(r0)
    imgr0.save('output_results/hrz1_16w_s2_label_bin.png')
    print("after scaling output")
    result_L = scale_intensity(label.numpy(), 8, 5.0)
    img_L = Image.fromarray(result_L)
    img_L.save('output_results/label_bin.png')


def test_data_generation():
    start = time.time()
    hdf5_train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train.h5"
    train_set = read_hdf(hdf5_train_path)

    hdf5_test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test.h5"
    test_set = read_hdf(hdf5_test_path)

    input_shape = (48, 48, 2)
    num_classes = 2

    seed = 0
    size = input_shape[0]
    stride = int(input_shape[0] / 4)
    # stride = 10

    break_tiles_info = (seed, size, stride)

    train_tiles_input, train_tiles_label = prepare_sliding_window_binary_data(read_hdf(hdf5_train_path), train_set,
                                                                              break_tiles_info, num_classes)
    test_tiles_input, test_tiles_label = prepare_sliding_window_binary_data(read_hdf(hdf5_test_path), test_set,
                                                                            break_tiles_info, num_classes)

    train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train_one-hrz-tiles_48w_12s_categorical_centered.h5"
    h5f = h5py.File(train_path, 'w')
    h5f.create_dataset('features', data=train_tiles_input, chunks=True)
    h5f.create_dataset('labels', data=train_tiles_label, chunks=True)
    h5f.close()

    test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test_one-hrz-tiles_48w_12s_categorical_centered.h5"
    h5f = h5py.File(test_path, 'w')
    h5f.create_dataset('features', data=test_tiles_input, chunks=True)
    h5f.create_dataset('labels', data=test_tiles_label, chunks=True)
    h5f.close()

    print(f"Total time for dataset generation: {time.time() - start}")


def test_train_from_generated_data():
    start = time.time()
    hdf5_train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train_one-hrz-tiles_64w_16s_categorical_centered.h5"

    train_h5_file = h5py.File(hdf5_train_path, 'r')
    train_tiles_input = np.array(train_h5_file.get('features'))
    train_tiles_label = np.array(train_h5_file.get('labels'))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_tiles_input, train_tiles_label))
    train_dataset = train_dataset.shuffle(buffer_size=10192).batch(batch_size=2048)

    train_h5_file.close()

    hdf5_test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test_one-hrz-tiles_64w_16s_categorical_centered.h5"

    test_h5_file = h5py.File(hdf5_test_path, 'r')
    test_tiles_input = np.array(test_h5_file.get('features'))
    test_tiles_label = np.array(test_h5_file.get('labels'))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_tiles_input, test_tiles_label))
    test_dataset = test_dataset.batch(batch_size=512)

    test_h5_file.close()

    input_shape = (16, 16, 2)
    num_classes_output = 2
    model = unet_tf2(input_shape=input_shape, output_channels=num_classes_output)

    model_name = "trained_models/model_64w_s16_bin_bal_centered_one-hrz-tiles"
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_metrics = {'train_mean_iou': MeanIoU(), 'train_cat_acc': tf.keras.metrics.CategoricalAccuracy()}
    val_metrics = {'val_mean_iou': MeanIoU(), 'val_cat_acc': tf.keras.metrics.CategoricalAccuracy()}
    metrics = (train_metrics, val_metrics)
    model = training_loop(train_dataset, test_dataset, model, model_name, optimizer, loss_fn, metrics, epochs=10)
    model.save(f'{model_name}.h5')

    print(f"Total training time: {time.time() - start}")


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # main()
    # test_data_generation()
    # test_train_from_generated_data()
    # test_training_loop()
    test_run_inside_sliding_window_with_seed_tile()
    # test_sliding_through_label()
