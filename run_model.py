import argparse
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from rockml.data.adapter.seismic.segy import SEGYDataAdapter
from rockml.data.transformations.seismic.image import Crop2D, ScaleIntensity, Transformation
from seisfast.io.horizon import Writer

import utils


def load_model(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    print("Loaded model from disk")

    return model


def get_input_seed(label_img: tf.Tensor,
                   canvas: tf.Tensor,
                   seed_class: int) -> Tuple[int, tf.Variable]:
    input_seed = 0
    for i in range(label_img.shape[0]):
        el = label_img[i, 8]
        if el == seed_class:
            input_seed = i
            break
    canvas_tile = tf.Variable(canvas)
    canvas_tile[input_seed, 0].assign(1)
    return input_seed, canvas_tile


def get_prediction_tiles(predictions: np.ndarray) -> np.ndarray:
    return np.argmax(predictions, axis=3)


def run_model(canvas: tf.Tensor,
              model: tf.keras.Model,
              inline_datum: Tuple[tf.Tensor, tf.Tensor],
              break_tiles_info) -> np.ndarray:
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


def predict_line(model: tf.keras.Model,
                 inference_config: dict,
                 segy: SEGYDataAdapter,
                 line_number: int,
                 crop: Transformation,
                 scale: Transformation,
                 num_horizons: int) -> Tuple[np.ndarray, np.ndarray]:
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
        input_seed, canvas = get_input_seed(label_img, canvas, seed_class)
        canvas = utils.clean_label(canvas)

        seed = input_seed - int(size / 2)
        break_tiles_info = (seed, size, stride)

        if input_seed > 0:
            output_horizon = run_model(canvas, model, inline_datum, break_tiles_info)
            horizon_list[horizon_number] = output_horizon.astype(np.uint8) * 175

    raw_amplitudes = seismic_datum.features
    clean_horizons = [i if i is not None else np.zeros([feat.shape[0], feat.shape[1]]) for i in horizon_list]
    horizons = np.stack(clean_horizons)

    return raw_amplitudes, horizons


def export_horizons(segy: SEGYDataAdapter,
                    horizons: np.array,
                    path: str) -> None:
    for cls in horizons.keys():
        writer = Writer(os.path.join(path, f'{cls}.xyz'))
        writer.write('inlines', np.asarray(horizons[cls], dtype=np.float32), segy.segy_raw_data)


def get_horizons_from_max(line_number: int,
                          amplitudes: np.array,
                          horizons: np.array,
                          crop_left: int,
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


def predict(model_path: str,
            segy_path: str,
            horizons_path_list: List[str],
            output_dir: str) -> None:
    # Load Model
    model = load_model(model_path)

    # Window configuration for inference
    inference_config = {}
    input_shape = (64, 64, 2)
    inference_config['size'] = input_shape[0]
    inference_config['stride'] = int(input_shape[0] / 8)

    # Load SEGY
    segy = SEGYDataAdapter(segy_path=segy_path,
                           horizons_path_list=horizons_path_list,
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
    utils.makedir(output_dir)

    # Loop through inlines
    for line_number in range(first_inline, last_inline + 1):
        print(f"Processing inline: {line_number}")

        # Run prediction through number of horizons
        amplitudes, horizons = predict_line(model, inference_config, segy, line_number,
                                            crop_transformation, scale_transformation, num_horizons)

        # Write .xyz files
        horizon_list = get_horizons_from_max(line_number, amplitudes, horizons, crop_left, crop_top)
        export_horizons(segy, horizon_list, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        help='Path to where trained model is.')
    parser.add_argument('--segy_path', type=str,
                        help='Path to where the segy is.')
    parser.add_argument('--horizons_path_list', type=str, nargs='+',
                        help='Path to the directory containing the input SEGY.')
    parser.add_argument('--output_path', type=str,
                        help='Directory where to save output SEGY.')
    parser.add_argument('--gpu_id', type=int,
                        help='Which GPU TensorFlow will use.')

    params = parser.parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[params.gpu_id], 'GPU')

    predict(params.model_path, params.segy_path, params.horizons_path_list, params.output_path)
