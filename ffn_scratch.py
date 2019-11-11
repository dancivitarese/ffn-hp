import os
import time
import h5py
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from rockml.data.adapter.seismic.segy import SEGYDataAdapter
from rockml.data.transformations.seismic.image import Crop2D, ScaleIntensity
from rockml.utils import io
from seisfast.io.horizon import Writer
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


def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    print("Loaded model from disk")

    return model


def clean_label(label_tile):
    label_tile = label_tile.numpy()
    non_zeros = label_tile > 0
    label_tile[non_zeros] = 1

    label_tile = tf.convert_to_tensor(label_tile, tf.uint8)

    return label_tile


def get_prediction_tiles(predictions):
    return np.argmax(predictions, axis=3)


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

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            epoch_loss += loss_value

            # Update training metric.
            for metric in train_metrics.values():
                metric(y_batch_train, logits)

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

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            # val_loss_value = loss_fn(y_batch_val, val_logits)
            # Update val metrics
            for metric in val_metrics.values():
                metric(y_batch_val, val_logits)

        for metric_name, metric in val_metrics.items():
            print(f"| {metric_name}: {metric.result()} ", end="", flush=True)
            metric.reset_states()
        print("|")
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

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        self.tf_mean_iou.update_state(y_true, y_pred)

    def result(self):
        return self.tf_mean_iou.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.tf_mean_iou.reset_states()


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


def run_model(canvas, model, inline_datum, break_tiles_info):
    seed = break_tiles_info[0]
    size = break_tiles_info[1]
    stride = break_tiles_info[2]

    feat, label = inline_datum
    feat = tf.reshape(feat, [feat.shape[0], feat.shape[1]])

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

        next_v_center = np.argmax(canvas_tile[:, h_seed + int(size / 2) + stride] > 0) + 1
        if np.abs(old_v_center - next_v_center) <= size:
            v_seed = next_v_center - int(size / 2)
            old_v_center = next_v_center

        if v_seed < 0:
            v_seed = 0
        elif v_seed > feat.shape[0] + size:
            v_seed = feat.shape[0] - size

    return canvas_tile.numpy()


def get_input_seed(label_img, canvas, seed_class, size):
    input_seed = 0
    for i in range(label_img.shape[0]):
        el = label_img[i, 8]
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
    stride = int(input_shape[0] / 8)

    model = load_model("trained_models/model_64w_s8_bin_bal_centered_one-hrz-tiles_30ep.h5")

    hdf5_valid_path = "/Users/gabriel/Documents/Dataset/dataset-10/valid.h5"
    valid_dataset = read_hdf(hdf5_valid_path)

    _, label_img = next(read_hdf(hdf5_valid_path))
    canvas = tf.zeros((400, 951), tf.uint8)
    seed_class = 7

    input_seed, canvas = get_input_seed(label_img, canvas, seed_class, size)
    canvas = clean_label(canvas)

    seed = input_seed - int(size / 2)

    break_tiles_info = (seed, size, stride)

    output = run_centering_sliding_window_binary(canvas, model, valid_dataset, break_tiles_info)

    feat, label = next(read_hdf(hdf5_valid_path))
    feat_img = Image.fromarray(tf.reshape(feat, [400, 951]).numpy())
    feat_img.save('output_results/feat_img.png')
    # label_img = Image.fromarray(label.numpy())
    # label_img.save('a_label_img.png')
    # output_img = Image.fromarray(output.astype(np.uint8))
    # output_img.save('a_output_image.png')
    print("before scaling output")
    # result_O = scale_intensity(output.astype(np.uint8), 8, 5.0)
    # img_O = Image.fromarray(result_O)x
    # img_O.save('output_results/s16w_h7_pred_fill.png')
    r0 = output.astype(np.uint8) * 175
    imgr0 = Image.fromarray(r0)
    imgr0.save('output_results/hrz7_64w_s8_seed_bin_bal_centered_one-hrz-tiles_window_gpu_30ep.png')
    print("after scaling output")
    result_L = scale_intensity(label.numpy(), 8, 5.0)
    img_L = Image.fromarray(result_L)
    img_L.save('output_results/label_fill.png')


def test_data_generation():
    start = time.time()
    hdf5_train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train.h5"
    train_set = read_hdf(hdf5_train_path)

    hdf5_test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test.h5"
    test_set = read_hdf(hdf5_test_path)

    input_shape = (64, 64, 2)
    num_classes = 2

    seed = 0
    size = input_shape[0]
    stride = int(input_shape[0] / 8)

    break_tiles_info = (seed, size, stride)

    train_tiles_input, train_tiles_label = prepare_sliding_window_binary_data(read_hdf(hdf5_train_path), train_set,
                                                                              break_tiles_info, num_classes)
    test_tiles_input, test_tiles_label = prepare_sliding_window_binary_data(read_hdf(hdf5_test_path), test_set,
                                                                            break_tiles_info, num_classes)

    train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train_one-hrz-tiles_64w_8s_categorical_centered.h5"
    h5f = h5py.File(train_path, 'w')
    h5f.create_dataset('features', data=train_tiles_input, chunks=True)
    h5f.create_dataset('labels', data=train_tiles_label, chunks=True)
    h5f.close()

    test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test_one-hrz-tiles_64w_8s_categorical_centered.h5"
    h5f = h5py.File(test_path, 'w')
    h5f.create_dataset('features', data=test_tiles_input, chunks=True)
    h5f.create_dataset('labels', data=test_tiles_label, chunks=True)
    h5f.close()

    print(f"Total time for dataset generation: {time.time() - start}")


def test_train_from_generated_data():
    start = time.time()
    hdf5_train_path = "/Users/gabriel/Documents/Dataset/dataset-10/train_one-hrz-tiles_64w_8s_categorical_centered.h5"

    train_h5_file = h5py.File(hdf5_train_path, 'r')
    train_tiles_input = np.array(train_h5_file.get('features'))
    train_tiles_label = np.array(train_h5_file.get('labels'))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_tiles_input, train_tiles_label))
    train_dataset = train_dataset.shuffle(buffer_size=10192).batch(batch_size=2048)

    train_h5_file.close()

    hdf5_test_path = "/Users/gabriel/Documents/Dataset/dataset-10/test_one-hrz-tiles_64w_8s_categorical_centered.h5"

    test_h5_file = h5py.File(hdf5_test_path, 'r')
    test_tiles_input = np.array(test_h5_file.get('features'))
    test_tiles_label = np.array(test_h5_file.get('labels'))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_tiles_input, test_tiles_label))
    test_dataset = test_dataset.batch(batch_size=512)

    test_h5_file.close()

    input_shape = (64, 64, 2)
    num_classes_output = 2
    model = unet_tf2(input_shape=input_shape, output_channels=num_classes_output)

    model_name = "trained_models/model_64w_s8_bin_bal_centered_one-hrz-tiles_30ep"
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_metrics = {'train_mean_iou': MeanIoU(), 'train_cat_acc': tf.keras.metrics.CategoricalAccuracy()}
    val_metrics = {'val_mean_iou': MeanIoU(), 'val_cat_acc': tf.keras.metrics.CategoricalAccuracy()}
    metrics = (train_metrics, val_metrics)
    model = training_loop(train_dataset, test_dataset, model, model_name, optimizer, loss_fn, metrics, epochs=30)
    model.save(f'{model_name}.h5')

    print(f"Total training time: {time.time() - start}")


def write_xyz():
    line_number = 253
    crop_left = 0
    crop_top = 63
    path = 'output_results/f3_horizons'
    amplitudes, horizons = temp_load_image()
    segy = SEGYDataAdapter(segy_path='/Users/gabriel/Documents/Dataset/netherlands_f3/seismic/F3_Netherlands.sgy',
                           horizons_path_list=[
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/North_Sea_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Chalk_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Rijnland_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/SSN_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Altena_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Germanic_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Zechstein_Group.txt"],
                           data_dict={})
    # Extracts
    horizon_list = get_horizons_from_max(line_number, amplitudes, horizons, crop_left, crop_top)
    io.makedir(path)
    export_horizons(segy, horizon_list, path)
    print('END')


def save_hrzs():
    path = 'output_results/f3_horizons'

    amplitudes, horizons = temp_load_image()

    dict_list = get_max(amplitudes, horizons)

    for i, _ in enumerate(dict_list):
        hrz = dict_list[i]
        figure = np.zeros(amplitudes.shape, dtype=np.uint8)
        for key, value in hrz.items():
            figure[value, key] = 174

        Image.fromarray(figure).save(f"{path}/hrz_{i + 1}.png")


def get_max(amplitudes, horizons):
    dict_list = []
    for horizon in horizons:
        columns, rows = np.where(np.transpose(horizon))

        p_dict = dict()
        amp = 0
        for row, column in enumerate(columns):
            if p_dict.get(column) is None:
                p_dict[column] = rows[row]
            elif amplitudes[rows[row], column] > amp:
                p_dict[column] = rows[row]
                amp = amplitudes[rows[row], column]

        dict_list += [p_dict]

    return dict_list


def export_horizons(segy: SEGYDataAdapter, horizons: np.array, path: str) -> None:
    for cls in horizons.keys():
        writer = Writer(os.path.join(path, f'{cls}.xyz'))
        writer.write('inlines', np.asarray(horizons[cls], dtype=np.float32), segy.segy_raw_data)


def get_horizons_from_max(line_number: int, amplitudes: np.array, horizons: np.array, crop_left: int,
                          crop_top: int) -> dict:
    horizon_dict = {f'hrz_{i}': [None] * horizons.shape[2] for i in range(1, horizons.shape[0] + 1)}

    bad_keys = []

    for idx, hrz_key in enumerate(horizon_dict.keys()):
        if np.sum(horizons[idx]) == 0:
            bad_keys.append(hrz_key)
            continue

        columns, rows = np.where(np.transpose(horizons[idx]))

        amp = 0
        for row, column in enumerate(columns):
            if horizon_dict[hrz_key][column] is None:
                horizon_dict[hrz_key][column] = [line_number, column + crop_left, rows[row] + crop_top]
            elif amplitudes[rows[row], column] > amp:
                horizon_dict[hrz_key][column] = [line_number, column + crop_left, rows[row] + crop_top]
                amp = amplitudes[rows[row], column]

        # Clean list up from None values (discontinuities)
        horizon_dict[hrz_key] = [rec for rec in horizon_dict[hrz_key] if rec]

    for bad_key in bad_keys:
        del horizon_dict[bad_key]

    return horizon_dict


def temp_load_image():
    raw = np.asarray(Image.open('output_results/feat_img.png'))
    h1 = np.asarray(Image.open(
        'output_results/hrz_1_in253.png'))
    h2 = np.asarray(Image.open(
        'output_results/hrz_2_in253.png'))
    h3 = np.asarray(Image.open(
        'output_results/hrz_3_in253.png'))
    h4 = np.asarray(Image.open(
        'output_results/hrz_4_in253.png'))
    h5 = np.asarray(Image.open(
        'output_results/hrz_5_in253.png'))
    h6 = np.asarray(Image.open(
        'output_results/hrz_6_in253.png'))
    h7 = np.asarray(Image.open(
        'output_results/hrz_7_in253.png'))

    hrz = np.stack((h1, h2, h3, h4, h5, h6, h7), axis=0)

    return raw, hrz


def test_run_from_specified_inline():
    input_shape = (64, 64, 2)

    size = input_shape[0]
    stride = int(input_shape[0] / 8)

    model = load_model("trained_models/model_64w_s8_bin_bal_centered_one-hrz-tiles_30ep.h5")

    segy = SEGYDataAdapter(segy_path='/Users/gabriel/Documents/Dataset/netherlands_f3/seismic/F3_Netherlands.sgy',
                           horizons_path_list=[
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/North_Sea_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Chalk_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Rijnland_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/SSN_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Altena_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Germanic_Group.txt",
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Zechstein_Group.txt"],
                           data_dict={'inlines': [[150, 300]]})

    seismic_datum = segy.get_line('inlines', 253)
    crop = Crop2D(0, 0, 63, 0)
    scale = ScaleIntensity(256, 5.0)
    seismic_datum = scale(crop(seismic_datum))
    feat = tf.dtypes.cast(seismic_datum.features, tf.uint8, name=None)
    label_img = tf.dtypes.cast(seismic_datum.label, tf.uint8, name=None)
    valid_dataset = iter([(feat, label_img)])

    # Show Feat
    Image.fromarray(tf.reshape(feat, [400, 951]).numpy()).show()

    # Show Label
    Image.fromarray(tf.reshape(label_img, [400, 951]).numpy()).show()

    canvas = tf.zeros((400, 951), tf.uint8)
    seed_class = 1

    input_seed, canvas = get_input_seed(label_img, canvas, seed_class, size)
    canvas = clean_label(canvas)

    seed = input_seed - int(size / 2)

    break_tiles_info = (seed, size, stride)

    output = run_centering_sliding_window_binary(canvas, model, valid_dataset, break_tiles_info)

    feat_img = Image.fromarray(tf.reshape(feat, [400, 951]).numpy())
    feat_img.save('output_results/feat_img.png')
    # label_img = Image.fromarray(label.numpy())
    # label_img.save('a_label_img.png')
    # output_img = Image.fromarray(output.astype(np.uint8))
    # output_img.save('a_output_image.png')
    print("before scaling output")
    # result_O = scale_intensity(output.astype(np.uint8), 8, 5.0)
    # img_O = Image.fromarray(result_O)x
    # img_O.save('output_results/s16w_h7_pred_fill.png')
    r0 = output.astype(np.uint8) * 175
    imgr0 = Image.fromarray(r0)
    imgr0.save('output_results/hrz_1_in253.png')
    print("after scaling output")
    result_L = scale_intensity(label_img.numpy(), 8, 5.0)
    img_L = Image.fromarray(result_L)
    img_L.save('output_results/label_fill.png')


def run_through_specified_inline(model, inference_config, segy, line_number, crop, scale, num_horizons):
    size = inference_config['size']
    stride = inference_config['stride']

    seismic_datum = segy.get_line('inlines', line_number)
    seismic_datum = scale(crop(seismic_datum))

    feat = tf.dtypes.cast(seismic_datum.features, tf.uint8, name=None)
    label_img = tf.dtypes.cast(seismic_datum.label, tf.uint8, name=None)
    inline_datum = (feat, label_img)

    horizon_list = [None] * num_horizons

    for horizon_number in range(num_horizons):
        seed_class = horizon_number
        canvas = tf.zeros(label_img.shape, tf.uint8)
        input_seed, canvas = get_input_seed(label_img, canvas, seed_class, size)
        canvas = clean_label(canvas)

        seed = input_seed - int(size / 2)
        break_tiles_info = (seed, size, stride)

        if input_seed > 0:
            output_horizon = run_model(canvas, model, inline_datum, break_tiles_info)
            horizon_list[horizon_number] = output_horizon.astype(np.uint8) * 175

    raw_amplitudes = seismic_datum.features
    clean_horizons = [i if i is not None else np.zeros([feat.shape[0], feat.shape[1]]) for i in horizon_list]
    horizons = np.stack(clean_horizons)

    return raw_amplitudes, horizons


def run_model_through_cube(args):
    # Load Model
    model = load_model(args.model_path)

    # Window configuration for inference
    inference_config = {}
    input_shape = (64, 64, 2)
    inference_config['size'] = input_shape[0]
    inference_config['stride'] = int(input_shape[0] / 8)

    # Load SEGY
    segy = SEGYDataAdapter(segy_path=args.segy_path,
                           horizons_path_list=eval(args.horizons_path_list),
                           data_dict={'inlines': [[150, 300]]})

    # Get number of inlines
    scan_result = segy.initial_scan()
    first_inline, last_inline = scan_result['range_inlines']

    # Number of horizons
    num_horizons = 7

    # Seismic transformations
    crop_left = 0
    crop_right = 0
    crop_top = 63
    crop_bottom = 0
    crop_transformation = Crop2D(crop_left, crop_right, crop_top, crop_bottom)
    scale_transformation = ScaleIntensity(256, 5.0)

    # Output Path
    io.makedir(args.output_dir)

    # Loop through inlines
    for line_number in range(first_inline, last_inline + 1):
        print(f"Processing inline: {line_number}")

        # Run prediction through number of horizons
        amplitudes, horizons = run_through_specified_inline(model, inference_config, segy, line_number,
                                                            crop_transformation, scale_transformation, num_horizons)

        # Write .xyz files
        horizon_list = get_horizons_from_max(line_number, amplitudes, horizons, crop_left, crop_top)
        export_horizons(segy, horizon_list, args.output_dir)


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

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default="trained_models/model_64w_s8_bin_bal_centered_one-hrz-tiles_30ep.h5",
                        help='Path to where trained model is.')
    parser.add_argument('--segy_path', type=str,
                        default='/Users/gabriel/Documents/Dataset/netherlands_f3/seismic/F3_Netherlands.sgy',
                        help='Path to where the segy is.')
    parser.add_argument('--horizons_path_list', type=str, default='[\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/North_Sea_Group.txt",\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Chalk_Group.txt",\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Rijnland_Group.txt",\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/SSN_Group.txt",\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Altena_Group.txt",\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Germanic_Group.txt",\
                               "/Users/gabriel/Documents/Dataset/netherlands_f3/horizons/Zechstein_Group.txt"]',
                        help='Path to the directory containing the input SEGY.')
    parser.add_argument('--output_dir', type=str, default="output_results/neth-horizons_f3_2",
                        help='Directory where to save output SEGY.')

    args = parser.parse_args()
    # test_data_generation()
    # test_train_from_generated_data()
    # test_run_inside_sliding_window_with_seed_tile()
    # test_run_from_specified_inline()
    # write_xyz()
    # save_hrzs()
    run_model_through_cube(args)
