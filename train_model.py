import argparse
import time
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf

from models.model_zoo import unet_tf2


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


def training_loop(train_dataset: tf.data.Dataset,
                  val_dataset: tf.data.Dataset,
                  model: tf.keras.Model,
                  model_path: str,
                  optimizer: tf.keras.optimizers.Optimizer,
                  loss_fn: tf.keras.losses.Loss,
                  metrics: Tuple[dict, dict],
                  epochs: int):
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

        model.save(f'{model_path}_{epoch}.h5')
        is_best = False
        if is_best:
            model.save(model_path)

    return model


def train_model(hdf5_train_path: str,
                hdf5_test_path: str,
                model_path: str,
                loss_fn: tf.keras.losses.Loss):
    """

    Args:
        hdf5_train_path:
        hdf5_test_path:
        model_path:
        loss_fn: Example: tf.keras.losses.CategoricalCrossentropy()

    Returns:

    """
    start = time.time()

    train_h5_file = h5py.File(hdf5_train_path, 'r')
    train_tiles_input = np.array(train_h5_file.get('features'))
    train_tiles_label = np.array(train_h5_file.get('labels'))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_tiles_input, train_tiles_label))
    train_dataset = train_dataset.shuffle(buffer_size=10192).batch(batch_size=2048)

    train_h5_file.close()

    test_h5_file = h5py.File(hdf5_test_path, 'r')
    test_tiles_input = np.array(test_h5_file.get('features'))
    test_tiles_label = np.array(test_h5_file.get('labels'))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_tiles_input, test_tiles_label))
    test_dataset = test_dataset.batch(batch_size=512)
    test_h5_file.close()

    input_shape = (64, 64, 2)
    num_classes_output = 2
    model = unet_tf2(input_shape=input_shape, output_channels=num_classes_output)

    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    train_metrics = {'train_mean_iou': MeanIoU(), 'train_cat_acc': tf.keras.metrics.CategoricalAccuracy()}
    val_metrics = {'val_mean_iou': MeanIoU(), 'val_cat_acc': tf.keras.metrics.CategoricalAccuracy()}
    metrics = (train_metrics, val_metrics)

    model = training_loop(train_dataset, test_dataset, model, model_path, optimizer, loss_fn, metrics, epochs=30)
    model.save(f'{model_path}.h5')

    print(f"Total training time: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_db_path', type=str,
                        help='Path to HDF5 training dataset.')
    parser.add_argument('--test_db_path', type=str,
                        help='Path to HDF5 test dataset.')
    parser.add_argument('--model_path', type=str,
                        help='Directory where to save trained model.')
    parser.add_argument('--gpu_id', type=int,
                        help='Which GPU TensorFlow will use.')

    args = parser.parse_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')

    train_model(hdf5_train_path=args.train_db_path,
                hdf5_test_path=args.test_db_path,
                model_path=args.model_path,
                loss_fn=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE))
