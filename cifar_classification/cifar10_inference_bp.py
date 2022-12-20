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
_WEIGHT_DECAY = 0

FLAGS = { "layers": 0 }


################################################################################
# With local connections, no weight sharing
# 1. mag 2
# 2. remove input bias
# 3. remove hidden bias
# 4. conv
################################################################################

_BATCH_NORM_DECAY = 0.995
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def fc_layer(inputs, filters):
    input_shape = inputs.get_shape().as_list()
    channels = input_shape[1]
    weight_initializer = tf.variance_scaling_initializer()
    weight_value = weight_initializer([channels , filters]) * 1e-2
    weight = tf.Variable(weight_value, name='label_filter_%d' % FLAGS["layers"], trainable=True)
    outputs = tf.matmul(inputs, weight)

    FLAGS["layers"] = FLAGS["layers"] + 1
    return outputs


def mask_kernel(kernel, mask, input_dimension, channels, output_dimension, filters):
    # return kernel
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


def conv_layer(inputs, kernel_size, filters, is_training):
    input_shape = inputs.get_shape().as_list()
    channels = input_shape[-1]
    output_size = input_shape[1]

    input_dimension = input_shape[1] * input_shape[2] * channels
    output_dimension = output_size * output_size * filters

    # mask
    mask = generate_mask(kernel_size, input_shape[1], input_shape[2])

    weight_initializer = tf.variance_scaling_initializer()
    kernel_value = weight_initializer([input_dimension, output_dimension]) * 1e-2
    kernel_value = mask_kernel(kernel_value, mask, input_shape[1] * input_shape[2], channels,
                               output_size * output_size, filters)
    # image part
    kernel = tf.Variable(kernel_value, name="filter_%d" % FLAGS["layers"], trainable=True)
    kernel = mask_kernel(kernel, mask, input_shape[1] * input_shape[2], channels,
                         output_size * output_size, filters)
    inputs = tf.reshape(inputs, [-1, input_dimension])
    outputs = tf.matmul(inputs, kernel)

    FLAGS["layers"] = FLAGS["layers"] + 1
    outputs = batch_norm_relu(outputs, is_training)
    # outputs = tf.nn.relu(outputs)
    return tf.reshape(outputs, [-1, output_size, output_size, filters])


def activation_fun(inputs):
    inputs = tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1,2,3])
    return inputs


def inference(images, feature_layers, kernel_size, channels, is_training):
    inputs = images
    for layer in range(feature_layers):
        inputs = conv_layer(inputs, kernel_size, channels, is_training)

    input_shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
    inputs = fc_layer(inputs, 10)
    return inputs


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.cast(labels, dtype=tf.int64)
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)


def training_sgd(loss, global_step, learning_rate):
    """Sets up the training Ops.
    """
    # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    variables = tf.trainable_variables()
    loss1 = loss + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in variables])
    # top_variables = [var for var in variables if var.name.find("dense_layer") >= 0]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = None
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss1, global_step=global_step, var_list=variables)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
    # Return the number of true entries.
    return tf.reduce_sum(input_tensor=tf.cast(correct, tf.int32))