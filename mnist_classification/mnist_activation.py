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


def normalized(images, labels):
    inputs = tf.cast(images, dtype=tf.float32)
    inputs_mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=1, keep_dims=True))
    inputs = tf.divide(inputs, tf.maximum(inputs_mag, 10e-8))
    inputs = tf.concat([inputs, labels], axis=1)
    return inputs


def inference(images, labels, feature_layers, learning_factor):
    labels = tf.one_hot(labels, depth=NUM_CLASSES, on_value=1.0, off_value=0.0, axis=-1)
    labels = tf.divide(labels, tf.sqrt(tf.constant(PIXEL_PER_CLASS, tf.float32)))
    labels = tf.tile(labels, [1, PIXEL_PER_CLASS])

    inputs = normalized(images, labels)
    layers = 0
    activations = []
    with tf.compat.v1.name_scope('hidden_layers'):
        for layer_i in range(feature_layers):
            weights = tf.Variable(
                tf.random.truncated_normal(
                    [LAYER_PIXELS, LAYER_PIXELS],
                    mean=0.0,
                    stddev=0.01
                ), name='weights')
            inputs = fc_layer(inputs, weights, learning_factor)
            if layer_i < feature_layers - 1:
                inputs = nonlinear_fun(inputs)
            activation = tf.reduce_sum(tf.pow(inputs, 2.0), axis=1)
            activations.append(activation)
            layers = layers + 1
    return activations


def fc_layer(x, w, learning_factor):
    output = tf.matmul(x, w)

    shape = tf.shape(w)
    input_dim = shape[0]
    classes = shape[1]
    u = tf.matmul(output, tf.transpose(w, [1, 0]))
    y = tf.tile(tf.reshape(output, [-1, 1, classes]), [1, input_dim, 1])
    z = tf.tile(tf.reshape(x - u, [-1, input_dim, 1]), [1, 1, classes])
    factor = learning_factor
    delta = tf.reduce_mean(tf.multiply(tf.transpose(tf.multiply(y, z), [1, 2, 0]), factor), axis=2)
    train_op = tf.assign(w, w + delta)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_op)
    return output


def nonlinear_fun2(inputs):
    return tf.abs(inputs)


def nonlinear_fun(inputs):
    mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=1, keepdims=True))
    inputs = tf.pow(tf.abs(inputs), 1.0)  # 0.75
    input_shape = inputs.get_shape().as_list()
    b = tf.Variable(tf.zeros([input_shape[1]], tf.float32), name="norm_mean", trainable=False)
    variance = tf.Variable(tf.ones([input_shape[1]], tf.float32) * 0.1, name="norm_variance", trainable=False)

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
    inputs = tf.divide(inputs, tf.maximum(tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1], keepdims=True)),
                                  1e-5))
    inputs = tf.multiply(inputs, mag)
    return inputs


def loss(logits, labels):
    labels = tf.cast(labels, dtype=tf.int64)
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

