"""Table both the images and their labels as the input
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
PIXEL_PER_CLASS = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
LABEL_PIXELS = NUM_CLASSES * PIXEL_PER_CLASS
LAYER_PIXELS = IMAGE_PIXELS + LABEL_PIXELS
HIDDEN_PIXELS = IMAGE_PIXELS
_WEIGHT_DECAY = 0

_BATCH_NORM_DECAY = 0.995
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def inference(inputs, feature_layers, is_training):
    with tf.compat.v1.name_scope('hidden_layers'):
        for layer_i in range(feature_layers):
            inputs = fc_layer(inputs, LAYER_PIXELS)
            inputs = batch_norm_relu(inputs, is_training)
    inputs = fc_layer(inputs, 10)
    return inputs


def fc_layer(inputs, filters):
    input_size = inputs.get_shape().as_list()[-1]
    weights = tf.Variable(
                tf.random.truncated_normal(
                    [input_size, filters],
                    mean=0.0,
                    stddev=0.01
                ), name='weights')
    inputs = tf.matmul(inputs, weights)
    return inputs


def loss(logits, labels):
    labels = tf.cast(labels, dtype=tf.int64)
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)


def training_sgd(loss, global_step, learning_rate):
    """Sets up the training Ops.
    """
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    variables = tf.trainable_variables()
    loss1 = loss + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in variables])
    # top_variables = [var for var in variables if var.name.find("dense_layer") >= 0]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss1, global_step=global_step, var_list=variables)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
    # Return the number of true entries.
    return tf.reduce_sum(input_tensor=tf.cast(correct, tf.int32))
