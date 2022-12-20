# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

NUM_CLASSES = 10
PIXEL_PER_CLASS = 10

FLAGS = { "layers": 0 }


################################################################################
# With local connections, no weight sharing
# 1. mag 2
# 2. remove input bias
# 3. remove hidden bias
# 4. conv
################################################################################

def fc_train(inputs, outputs, weight, learning_rate):
    u = tf.matmul(outputs, tf.transpose(weight, [1, 0]))
    delta = tf.matmul(tf.multiply(tf.transpose(inputs - u, [1, 0]), learning_rate), outputs)
    batch_size = tf.shape(outputs)[0]
    delta = tf.divide(delta, tf.cast(batch_size, tf.float32))
    # weight decay is only for no feedback!
    delta = delta - weight * 1e-5
    return delta


def fc_layer(inputs, filters, learning_rate):
    input_shape = inputs.get_shape().as_list()
    channels = input_shape[1]
    weight_initializer = tf.variance_scaling_initializer()
    weight_value = weight_initializer([channels , filters]) * 1e-1
    weight = tf.Variable(weight_value, name='label_filter_%d' % FLAGS["layers"], trainable=False)
    outputs = tf.matmul(inputs, weight)

    # for training
    delta_fc = fc_train(inputs, outputs, weight, learning_rate)
    new_weight = tf.add(weight, delta_fc)
    fc_op = tf.assign(weight, tf.maximum(tf.minimum(new_weight, 1.0), -1.0))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, fc_op)

    FLAGS["layers"] = FLAGS["layers"] + 1
    return outputs


def mask_kernel(kernel, mask, input_dimension, channels, output_dimension, filters):
    kernel = tf.reshape(kernel, [input_dimension, channels, output_dimension, filters])
    kernel = tf.transpose(kernel, [1, 3, 0, 2])
    kernel = tf.multiply(kernel, mask)
    kernel = tf.transpose(kernel, [2, 0, 3, 1])
    return tf.reshape(kernel, [input_dimension * channels, output_dimension * filters])


def generate_mask(kernel_size, input_shape1, input_shape2):
    mask = np.zeros([input_shape1 * input_shape2, input_shape1 * input_shape2], np.float)
    for a1 in range(input_shape1):
        for a2 in range(input_shape2):
            a = a1 * input_shape2 + a2
            for b1 in range(input_shape1):
                for b2 in range(input_shape2):
                    b = b1 * input_shape2 + b2
                    if (a1 - (kernel_size // 2) <= b1 <= a1 + (kernel_size // 2)) and (
                            a2 - (kernel_size // 2) <= b2 <= a2 + (kernel_size // 2)):
                        mask[a, b] = 1.0
    return tf.constant(mask, tf.float32)


def conv_layer(inputs, kernel_size, filters, learning_rate):
    input_shape = inputs.get_shape().as_list()
    channels = input_shape[-1]
    output_size = input_shape[1]

    input_dimension = input_shape[1] * input_shape[2] * channels
    output_dimension = output_size * output_size * filters

    # mask
    mask = generate_mask(kernel_size, input_shape[1], input_shape[2]) if kernel_size <= 32 else None

    weight_initializer = tf.variance_scaling_initializer()
    kernel_value = weight_initializer([input_dimension, output_dimension]) * 1e-1
    kernel_value = mask_kernel(kernel_value, mask, input_shape[1] * input_shape[2], channels,
                               output_size * output_size, filters) if kernel_size <= 32 else kernel_value
    # image part
    kernel = tf.Variable(kernel_value, name="filter_%d" % FLAGS["layers"], trainable=False)
    inputs = tf.reshape(inputs, [-1, input_dimension])
    outputs = tf.matmul(inputs, kernel)

    # for training
    delta = fc_train(inputs, outputs, kernel, learning_rate)
    new_kernel = tf.add(kernel, delta)
    new_kernel = mask_kernel(new_kernel, mask, input_shape[1] * input_shape[2], channels, output_size * output_size,
                             filters) if kernel_size <= 32 else new_kernel
    hebb_op = tf.assign(kernel, tf.maximum(tf.minimum(new_kernel, 1.0), -1.0))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, hebb_op)

    FLAGS["layers"] = FLAGS["layers"] + 1
    return tf.reshape(outputs, [-1, output_size, output_size, filters])


def input_layer(inputs, labels, kernel_size, filters, learning_rate):
    input_shape = inputs.get_shape().as_list()
    channels = input_shape[-1]
    output_size = input_shape[1]

    input_dimension = input_shape[1] * input_shape[2] * channels
    output_dimension = output_size * output_size * filters

    # mask
    mask = generate_mask(kernel_size, input_shape[1], input_shape[2]) if kernel_size <= 32 else None

    weight_initializer = tf.variance_scaling_initializer()
    kernel_value = weight_initializer([input_dimension, output_dimension]) * 1e-1
    kernel_value = mask_kernel(kernel_value, mask, input_shape[1] * input_shape[2], channels,
                               output_size * output_size, filters) if kernel_size <= 32 else kernel_value
    # image part
    kernel = tf.Variable(kernel_value, name="filter_%d" % FLAGS["layers"], trainable=False)
    inputs = tf.reshape(inputs, [-1, input_dimension])
    outputs = tf.matmul(inputs, kernel)

    # label part
    label_len = labels.get_shape().as_list()[-1]
    output_d = output_size * output_size * filters
    weight_value = weight_initializer([label_len, output_d]) * 1e-1
    weight = tf.Variable(weight_value, name = 'label_filter_%d' % FLAGS["layers"], trainable = False)
    label_outputs = tf.matmul(labels, weight)
    outputs = outputs + label_outputs

    # for training
    delta = fc_train(inputs, outputs, kernel, learning_rate)
    new_kernel = tf.add(kernel, delta)
    new_kernel = mask_kernel(new_kernel, mask, input_shape[1] * input_shape[2], channels,
                             output_size * output_size, filters) if kernel_size <= 32 else new_kernel
    hebb_op = tf.assign(kernel, tf.maximum(tf.minimum(new_kernel, 1.0), -1.0))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, hebb_op)

    delta_label = fc_train(labels, outputs, weight, learning_rate)
    new_weight = tf.add(weight, delta_label)
    label_op = tf.assign(weight, tf.maximum(tf.minimum(new_weight, 1.0), -1.0))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, label_op)

    FLAGS["layers"] = FLAGS["layers"] + 1
    return tf.reshape(outputs, [-1, output_size, output_size, filters])


def nonlinear_fun2(inputs):
    mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1,2,3], keepdims=True))
    inputs = tf.pow(tf.abs(inputs), 0.75) # 0.75
    bias = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
    inputs = inputs - bias
    inputs = tf.divide(inputs, tf.maximum(tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1,2,3], keepdims=True)), 1e-5))
    inputs = tf.multiply(inputs, mag)
    return inputs


def nonlinear_fun(inputs):
    mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1, 2, 3], keepdims=True))
    inputs = tf.pow(tf.abs(inputs), 1.0)  # 0.75
    input_shape = inputs.get_shape().as_list()
    b = tf.Variable(tf.zeros([input_shape[1], input_shape[2], input_shape[3]], tf.float32), name="norm_mean", trainable=False)
    variance = tf.Variable(tf.ones([input_shape[1], input_shape[2], input_shape[3]], tf.float32) * 0.1,
                           name="norm_variance", trainable=False)

    decay = 0.995
    m = tf.reduce_mean(inputs, axis=0)
    new_m = b * decay + m * (1.0 - decay)
    mean_op = tf.assign(b, new_m)
    v = tf.reduce_mean(tf.multiply(inputs - new_m, inputs - new_m), axis=0)
    new_v = variance * decay + v * (1.0 - decay)
    variance_op = tf.assign(variance, new_v)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_op)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_op)

    inputs = tf.divide(inputs - b, tf.sqrt(tf.maximum(variance, 1e-5)))
    inputs = tf.divide(inputs, tf.maximum(tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1, 2, 3], keepdims=True)),
                                  1e-5))
    inputs = tf.multiply(inputs, mag)

    return inputs


def activation_fun(inputs):
    inputs = tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1,2,3])
    return inputs


def inference(images, labels, feature_layers, kernel_size, channels, learning_rate):
    target_mag = 2.0
    images = tf.multiply(tf.divide(images, tf.sqrt(tf.reduce_sum(tf.multiply(images, images), axis=[1,2,3], keepdims=True))),
                         tf.sqrt(tf.constant(target_mag, tf.float32)))

    labels = tf.one_hot(labels, depth=NUM_CLASSES, on_value=1.0, off_value=0.0, axis=-1)
    labels = tf.tile(labels, [1, PIXEL_PER_CLASS])
    labels = tf.divide(labels, tf.sqrt(tf.reduce_sum(tf.multiply(labels, labels), axis=1, keepdims=True)))

    # 1st layer
    activations = []
    inputs = input_layer(images, labels, kernel_size, channels, learning_rate)
    activations.append(activation_fun(inputs))
    inputs = nonlinear_fun(inputs)

    # hidden layers
    for i in range(1, feature_layers - 1):
        inputs = conv_layer(inputs, kernel_size, channels, learning_rate)
        activations.append(activation_fun(inputs))
        inputs = nonlinear_fun(inputs)

    input_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
    inputs = fc_layer(inputs, input_shape[1] * input_shape[2] * channels, learning_rate)
    activation = tf.reduce_sum(tf.multiply(inputs, inputs), axis=1)
    activations.append(activation)

    return activations
