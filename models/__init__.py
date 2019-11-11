import math

import tensorflow as tf


class Danet(object):
    """ Danet base model. It has methods for creating a simple convolutional network as well as
        the ResNet model.
        ResNet related papers:
        https://arxiv.org/pdf/1603.05027v2.pdf
        https://arxiv.org/pdf/1512.03385v1.pdf
        https://arxiv.org/pdf/1605.07146v1.pdf
        Some code ideas from:
        https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator
    """

    def __init__(self, is_train, hparam):
        """ Danet base model constructor.
        Args:
            is_train: bool indicating if the model is going to be used for training or not.
            hparam: argparse Namespace with various hyperparameters.
        """
        self.is_train = is_train
        self.layer_info = None  # refer to calculate_rf()
        self.batch_norm_mu = hparam.batch_norm_mu
        self.batch_norm_epsilon = hparam.batch_norm_epsilon
        self.num_classes = hparam.dataset.num_classes
        self.image_height = hparam.dataset.tile_shape[0]
        self.image_width = hparam.dataset.tile_shape[1]
        self.image_channels = hparam.dataset.tile_shape[2]
        self.kernel_init = None
        self.bias_init = tf.zeros_initializer

    @staticmethod
    def _calculate_rf(kernel, stride, padding, layer_in):
        """ From https://gist.github.com/Nikasa1889/781a8eb20c5b32f8e378353cde4daa51
            [filter size, stride, padding]
            Assume the two dimensions are the same
            Each kernel requires the following parameters:
            - k_i: kernel size
            - s_i: stride
            - p_i: padding (if padding is uneven, right padding will higher than left padding;
                   "SAME" option in tensorflow)
            Each layer i requires the following parameters to be fully represented:
            - n_i: number of feature (data layer has n_1 = imagesize )
            - j_i: distance (projected to image pixel distance) between center of two adjacent
                   features
            - r_i: receptive field of a feature in layer i
        """
        n_in = layer_in[0]
        j_in = layer_in[1]
        r_in = layer_in[2]

        n_out = math.floor((n_in - kernel + 2 * padding) / stride) + 1
        j_out = j_in * stride
        r_out = r_in + (kernel - 1) * j_in
        return n_out, j_out, r_out

    @staticmethod
    def _adjust_shape(tensor, shape):
        """ This method fixes the size of output tensors from transposed convolutions with
            even sizes, e.g., conv_t of shape (7, 7) would be 14 instead of 13.
        Args:
            tensor: input tensor 4D (NHWC).
            shape: desired shape for the tensor as a list.
        Returns:
            output tensor
        """

        new_tensor = tensor
        tensor_shape = tensor.get_shape().as_list()
        diff = [tensor_shape[i] - shape[i] for i in [1, 2]]

        if diff[0] != 0:
            new_tensor = tensor[:, :-diff[0], :, :]

        if diff[1] != 0:
            new_tensor = new_tensor[:, :, :-diff[1], :]

        return new_tensor

    @staticmethod
    def _pad_channels(tensors):
        """ Pad with zeros channels of the tensor in *tensors* list that have the smaller number
            of feature maps.
        Args:
            tensors: list of 2 tensors to compare sizes.
        Returns:
            tensors with matching number of channels
        """

        shapes = [tensors[0].get_shape().as_list(), tensors[1].get_shape().as_list()]
        channel_axis = -1

        if shapes[0][channel_axis] < shapes[1][channel_axis]:
            small_ch_id, large_ch_id = (0, 1)
        else:
            small_ch_id, large_ch_id = (1, 0)

        pad = (shapes[large_ch_id][channel_axis] - shapes[small_ch_id][channel_axis])
        pad_beg = pad // 2
        pad_end = pad - pad_beg

        tensors[small_ch_id] = tf.pad(tensors[small_ch_id], [[0, 0], [0, 0], [0, 0],
                                                             [pad_beg, pad_end]])
        return tensors

    def _residual_v1(self, tensor, kernel_size, filters, strides, name=None):
        """ Residual unit with 2 sub layers, using Plan A for shortcut connection.
        Args:
            tensor: input tensor 4D (NHWC).
            kernel_size: (int) convolution kernel size.
            filters: (int) number of convolutional filters.
            strides: (int) convolution stride. This operation supports only strides equal
                to 1 or 2. If strides=1, the image size will not change. If strides=2,
                the image will be reduced by half.
            name: (str) name of the block.
        Returns:
            output tensor
        """
        with tf.name_scope('residual_v1') as name_scope:
            with tf.variable_scope(name) as scope:
                tf.logging.debug(f'Building residual block {name_scope}/{scope}.')
                orig_x = tensor

                tensor = tf.nn.relu(
                    self._batch_norm(self._residual_conv(inputs=tensor, kernel_size=kernel_size,
                                                         filters=filters, strides=strides)))

                tensor = self._batch_norm(self._residual_conv(tensor, kernel_size, filters, 1))

                if strides == 2:
                    orig_x = self._avg_pool(inputs=orig_x, pool_size=strides, strides=strides)

                orig_x, tensor = self._pad_channels([orig_x, tensor])
                tensor = tf.nn.relu(tf.add(tensor, orig_x))

                tf.logging.debug(f'Inputs shape after unit {name_scope}/{scope}: '
                                 f'{tensor.get_shape()}')
            return tensor

    def _residual_transposed_v1(self, tensor, kernel_size, filters, strides, name=None):
        """ Transposed residual unit with 2 sub layers, using Plan A for shortcut connection.
            Instead of using an unpooling operation to upscale the shortcut, we use a
            transposed convolution.
        Args:
            tensor: input tensor 4D (NHWC).
            kernel_size: (int) convolution kernel size.
            filters: (int) number of convolutional filters.
            strides: (int) convolution stride. This operation supports only strides equal
                to 1 or 2. If strides=1, the image size will not change. If strides=2,
                the image will be reduced by half.
            name: (str) name of the block.
        Returns:
            output tensor
        """
        with tf.name_scope('residual_transposed_v1') as name_scope:
            with tf.variable_scope(name) as scope:
                tf.logging.debug(f'Building transposed residual block {name_scope}/{scope}.')
                orig_x = tensor

                tensor = self._conv2d_transpose(inputs=tensor,
                                                kernel_size=kernel_size,
                                                filters=filters,
                                                strides=strides)
                tensor = self._batch_norm(tensor)
                tensor = tf.nn.relu(tensor)

                tensor = self._conv2d_transpose(inputs=tensor,
                                                kernel_size=kernel_size,
                                                filters=filters,
                                                strides=1)
                tensor = self._batch_norm(tensor)

                if strides == 2:
                    orig_x = self._conv2d_transpose(orig_x, kernel_size, filters, strides)

                tensor = tf.nn.relu(tf.add(tensor, orig_x))

                tf.logging.debug(f'Inputs shape after unit {name_scope}/{scope}: '
                                 f'{tensor.get_shape()}')

            return tensor

    def _batch_norm(self, inputs, name=None):
        """ Batch normalization wrapper.
            See:
                https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
                https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
        Args:
            inputs: input tensor (NHWC).
            name: (str) name of the layer.
        Returns:
            output tensor
        """
        return tf.layers.batch_normalization(inputs=inputs,
                                             axis=-1,
                                             momentum=self.batch_norm_mu,
                                             epsilon=self.batch_norm_epsilon,
                                             training=self.is_train,
                                             renorm=False,
                                             renorm_clipping=None,
                                             renorm_momentum=0.99,
                                             name=name)

    def _conv2d(self, inputs, kernel_size, filters, strides, activation=None,
                initializer=None, padding='SAME', name=None):
        """ Convolution operation wrapper. It also calculates the receptive field at this point.
        Args:
            inputs: input tensor.
            kernel_size: int representing the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: int specifying the strides of the convolution operation (1 means [1, 1]).
            activation: activation function. Set it to None to maintain a linear activation.
            initializer: an initializer for the convolution kernel.
            padding: One of *valid* or *same* (case-insensitive).
            name: name of the layer.
        Returns:
            output tensor
        """
        self.layer_info = self._calculate_rf(kernel=kernel_size,
                                             stride=strides,
                                             padding=kernel_size // strides,
                                             layer_in=self.layer_info)
        # tf.logging.debug(f'Layer {name}: RF {self.layer_info[-1]}, '
        #                  f'shape {inputs.shape}')

        return tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                activation=activation,
                                padding=padding,
                                strides=strides,
                                data_format='channels_last',
                                kernel_initializer=initializer,
                                bias_initializer=self.bias_init(), name=name)

    def _conv2d_transpose(self, inputs, kernel_size, filters, strides, activation=None,
                          initializer=None, padding='SAME', name=None):
        """ Transpose Convolution operation wrapper.
        Args:
            inputs: input tensor.
            kernel_size: int representing the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: int specifying the strides of the convolution operation (1 means [1, 1]).
            activation: activation function. Set it to None to maintain a linear activation.
            initializer: an initializer for the convolution kernel.
            padding: One of *valid* or *same* (case-insensitive).
            name: name of the layer.
        Returns:
            output tensor
        """

        tf.logging.debug(f'Layer {name}: shape {inputs.shape}')

        return tf.layers.conv2d_transpose(inputs=inputs,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          padding=padding,
                                          strides=strides,
                                          data_format='channels_last',
                                          kernel_initializer=initializer,
                                          bias_initializer=self.bias_init(),
                                          name=name)

    def _max_pooling2d(self, inputs, pool_size, strides, name=None):
        """ Max pooling operation wrapper. It also calculates the receptive field at this point.
        Args:
            inputs: input tensor. Must have rank 4.
            pool_size: int representing the size of the pooling window (3 means [3, 3]).
            strides: int specifying the strides of the pooling operation (1 means [1, 1]).
            name: (str) name for the layer.
        Returns:
            output tensor.
        """
        self.layer_info = self._calculate_rf(kernel=pool_size,
                                             stride=strides,
                                             padding=pool_size // strides,
                                             layer_in=self.layer_info)
        tf.logging.debug(f'Layer {name}: RF {self.layer_info[-1]}, '
                         f'shape {inputs.shape}')

        return tf.layers.max_pooling2d(inputs=inputs,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding='SAME',
                                       data_format='channels_last',
                                       name=name)

    def _residual_conv(self, inputs, kernel_size, filters, strides,
                       is_atrous=False, initializer=None, name=None):
        """ Convolution operation for residual unit wrapper. There is no bias and padding is
            determined by *is_atrous*. It also calculates the receptive field at this point.
        Args:
            inputs: input tensor.
            kernel_size: int representing the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: int specifying the strides of the convolution operation (1 means [1, 1]).
            is_atrous: bool indicating if the conv operation is atrous.
            initializer: an initializer for the convolution kernel.
            name: name of the layer.
        Returns:
            output tensor
        """
        padding = 'SAME'

        if not is_atrous and strides > 1:
            pad = kernel_size - 1
            pad_beg = pad // 2
            pad_end = pad - pad_beg
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            padding = 'VALID'

        self.layer_info = self._calculate_rf(kernel=kernel_size,
                                             stride=strides,
                                             padding=kernel_size // strides,
                                             layer_in=self.layer_info)

        tf.logging.debug('\tReceptive field: %s', self.layer_info[-1])

        return tf.layers.conv2d(inputs=inputs,
                                kernel_size=kernel_size,
                                filters=filters,
                                strides=strides,
                                padding=padding,
                                use_bias=False,
                                data_format='channels_last',
                                kernel_initializer=initializer,
                                name=name)

    def _fully_connected(self, inputs, units, activation=None, initializer=None, name=None):
        """ Dense layer operation wrapper.
        Args:
            inputs: input tensor.
            units: (int) dimensionality of the output space.
            activation: activation function. Set it to None to maintain a linear activation.
            initializer: an initializer for the convolution kernel.
            name: (str) name of the layer.
        Returns:
            output tensor
        """

        inputs = tf.layers.dense(inputs=inputs,
                                 units=units,
                                 activation=activation,
                                 kernel_initializer=initializer,
                                 bias_initializer=self.bias_init(),
                                 name=name)

        tf.logging.debug(f'Image after unit {name}: {inputs.get_shape()}')
        return inputs

    def _avg_pool(self, inputs, pool_size, strides, name=None):
        """ Avg pooling operation wrapper. It also calculates the receptive field at this point.
        Args:
            inputs: input tensor. Must have rank 4.
            pool_size: int representing the size of the pooling window (3 means [3, 3]).
            strides: int specifying the strides of the pooling operation (1 means [1, 1]).
            name: (str) name for the layer.
        Returns:
            output tensor.
        """
        inputs = tf.layers.average_pooling2d(inputs=inputs,
                                             pool_size=pool_size,
                                             strides=strides,
                                             data_format='channels_last',
                                             padding='SAME',
                                             name=name)

        tf.logging.debug('\tReceptive field: %s', self.layer_info[-1])
        tf.logging.debug(f'Image after unit {name}: {inputs.get_shape()}')
        return inputs

    def forward_pass(self, inputs):
        raise NotImplementedError('forward_pass() is implemented in Danet sub classes')
