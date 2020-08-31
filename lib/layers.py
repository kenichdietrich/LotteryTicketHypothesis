import numpy as np
import tensorflow as tf

import os
from typing import Callable, Tuple

""" LTH-adapted implementation of usual NN layers. """

class LotteryLinear(tf.keras.layers.Layer):
    """ Fully Connected layer adapted to use Lottery Ticket Hypothesis
    in neural networks with Keras + Tensorflow>=2.2.

    Parameters
    ----------
        units (int): Number of neurons in the layer.
        activation (tf.nn.function): Activation function for the outputs.
    """

    def __init__(
        self,
        units: int = 128,
        activation: Callable = tf.nn.relu
    ):
        super(LotteryLinear, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # We use normal Glorot initialization for the weights
        self.init_sd = np.sqrt(2 / (input_dim+self.units))

        k_init = tf.random_normal_initializer(stddev=self.init_sd)
        self.kernel = tf.Variable(
            initial_value = k_init(shape = (input_dim, self.units), 
                                   dtype = "float32"),
            trainable = True,
            name = "kernel"
        )
        # Bias initialization
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(
            initial_value = b_init(shape = (self.units,),
                                  dtype = "float32"),
            trainable = True,
            name = os.path.split(self.kernel.name)[-1] + "_bias"
        )
        # Untrainable mask to prune weights
        m_init = tf.ones_initializer()
        self.mask = tf.Variable(
            initial_value = m_init(shape = (input_dim, self.units),
                                  dtype = "float32"),
            trainable = False,
            name = os.path.split(self.kernel.name)[-1] + "_mask"
        )
        
    def call(self, inputs):
        # Applying the mask to the weights (element-wise product)
        masked_kernel = self.kernel*self.mask
        # Computing the outputs and activating them
        output = inputs@masked_kernel + self.bias
        output = self.activation(output)

        return output   


class LotteryConv2D(tf.keras.layers.Layer):
    """ 2D Convolutional layer adapted to use Lottery Ticket Hypothesis
    in neural networks with Keras + Tensorflow>=2.2.

    Parameters
    ----------
        kernel_size (tuple): Shape (2 dimensions) of the convolutional 
        kernel to be applied to the images.
        filters (int): Number of filters to be applied. They make up 
        the feature mapping.
        stride (tuple): Convolution stride.
        padding (str): "same" or "valid" padding algorithm.
        activation(tf.nn.function): Activation function for the 
        convoluted output images.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        filters: int = 3,
        stride: Tuple[int, int] = (1, 1),
        padding: str = "same",
        activation: Callable = tf.nn.relu
    ):
        super(LotteryConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.n_filters = filters
        self.stride = stride
        self.padding = padding.upper()
        self.activation = activation
    
    def build(self, input_shape):
        # Filters in TF convolution must have the format 
        # (kernel_rows, kernel_cols, input_channels, output_channels)
        filter_dims = self.kernel_size + (input_shape[-1], self.n_filters)
        # We use normal He initialization for the filter weights
        self.init_sd = np.sqrt(2 / np.array(input_shape[1:]).prod())

        f_init = tf.random_normal_initializer(stddev=self.init_sd)
        self.filters = tf.Variable(
            initial_value = f_init(shape = filter_dims, 
                                  dtype = "float32"),
            trainable = True
        )
        # Bias initialization
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(
            initial_value = b_init(shape = (1, 1, self.n_filters),
                                  dtype = "float32"),
            trainable = True,
            name = os.path.split(self.filters.name)[-1] + "_bias"
        )
        # Untrainable mask to prune filters
        m_init = tf.ones_initializer()
        self.mask = tf.Variable(
            initial_value = m_init(shape = filter_dims,
                                  dtype = "float32"),
            trainable = False,
            name = os.path.split(self.filters.name)[-1] + "_mask"
        )

    def call(self, inputs):
        # Applying the mask to the filters (element-wise product)
        masked_filters = self.filters*self.mask
        # Computing convolution
        conv = tf.nn.conv2d(input=inputs, 
                            filters=masked_filters, 
                            strides=self.stride, 
                            padding=self.padding, 
                            data_format="NHWC")
        # Adding bias to obtain the feature mapping
        feat_map = conv+self.bias 
        # Activating output feature mapping
        output = self.activation(feat_map)

        return output

            
            


        