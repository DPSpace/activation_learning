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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import random
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import mnist_activation as mnist

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.compat.v1.placeholder(
        tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    input_labels_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))
    labels_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, input_labels_placeholder, labels_placeholder


def one_hot(labels_feed):
    label_vector = np.eye(10)[labels_feed]
    label_vector = np.divide(label_vector, np.sqrt(mnist.PIXEL_PER_CLASS))
    # return np.pad(label_vector, ((0, 0), (0, (mnist.PIXEL_PER_CLASS - 1) * mnist.NUM_CLASSES)),  "constant")
    return np.tile(label_vector, [1, mnist.PIXEL_PER_CLASS])


def do_eval(sess,
            predicted_label,
            reset_op,
            update_op,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    if steps_per_epoch > 100:
        steps_per_epoch = 100
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        if step % 10 == 0:
            print(step)
        images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                       FLAGS.fake_data)
        feed_dict = {
            images_placeholder: images_feed,
            labels_placeholder: labels_feed
        }

        sess.run(reset_op, feed_dict=feed_dict)
        for i in range(100):
            sess.run(update_op, feed_dict=feed_dict)

        result = sess.run(predicted_label, feed_dict=feed_dict)
        # print(result)
        true_count += np.sum(np.equal(result, labels_feed))
    precision = float(true_count) / num_examples
    return precision


def normalize(images):
    inputs = tf.cast(images, dtype=tf.float32)
    inputs_mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=1, keep_dims=True))
    inputs = tf.divide(inputs, tf.maximum(inputs_mag, 10e-8))
    return inputs


def inference2(images, labels, feature_layers, learning_factor):
    labels = tf.one_hot(labels, depth=mnist.NUM_CLASSES, on_value=1.0, off_value=0.0, axis=-1)
    labels = tf.divide(labels, tf.sqrt(tf.constant(mnist.PIXEL_PER_CLASS, tf.float32)))
    labels = tf.tile(labels, [1, mnist.PIXEL_PER_CLASS])
    images = normalize(images)

    label_value = 1.0 / tf.sqrt(tf.constant(mnist.PIXEL_PER_CLASS, tf.float32))

    inputs = tf.concat([images, labels], axis=1)
    layers = 0
    activations = []
    layer_units = []
    with tf.compat.v1.name_scope("layer_units"):
        label_infer = tf.Variable(tf.zeros([FLAGS.batch_size, mnist.LABEL_PIXELS], tf.float32), name="label_infer")
        reset_op = tf.assign(label_infer, tf.zeros([FLAGS.batch_size, mnist.LABEL_PIXELS], tf.float32))
        tf.add_to_collection("reset_units", reset_op)
        for layer_i in range(feature_layers):
            layer_unit = tf.Variable(tf.zeros([FLAGS.batch_size, mnist.LAYER_PIXELS], tf.float32), name="layer_unit")
            layer_units.append(layer_unit)
            reset_op = tf.assign(layer_unit, tf.zeros([FLAGS.batch_size, mnist.LAYER_PIXELS], tf.float32))
            tf.add_to_collection("reset_units", reset_op)

    layer_w = []
    layer_b = []
    layer_variance = []
    with tf.compat.v1.name_scope('hidden_layers'):
        for layer_i in range(feature_layers):
            weights = tf.Variable(
                tf.random.truncated_normal(
                    [mnist.LAYER_PIXELS, mnist.LAYER_PIXELS],
                    mean=0.0,
                    stddev=0.01
                ), name='weights')
            layer_w.append(weights)
            inputs = tf.matmul(inputs, weights)
            if layer_i < feature_layers - 1:
                mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=1, keepdims=True))
                inputs = tf.pow(tf.abs(inputs), 1.0)  # 0.75
                input_shape = inputs.get_shape().as_list()
                b = tf.Variable(tf.zeros([input_shape[1]], tf.float32), name="norm_mean", trainable=False)
                variance = tf.Variable(tf.ones([input_shape[1]], tf.float32) * 0.1, name="norm_variance",
                                       trainable=False)
                layer_b.append(b)
                layer_variance.append(variance)
                inputs = tf.divide(inputs - b, tf.sqrt(tf.maximum(variance, 1e-5)))
                inputs = tf.divide(inputs, tf.maximum(
                    tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1], keepdims=True)),
                    1e-5))
                inputs = tf.multiply(inputs, mag)
            activation = tf.reduce_sum(tf.pow(inputs, 2.0), axis=1)
            activations.append(activation)
            layers = layers + 1

    with tf.compat.v1.name_scope('layer_propagation'):
        layer_messages = []
        for layer_i in range(feature_layers):
            # forward
            layer_x = tf.concat([images, label_infer], axis=1) if layer_i == 0 else layer_units[layer_i-1]
            layer_y = tf.matmul(layer_x, layer_w[layer_i])
            if layer_i < feature_layers - 1:
                layer_y_sign = tf.cast(tf.greater(layer_y, 0.0), tf.float32) * 2.0 - 1.0
                layer_y_positive = tf.abs(layer_y)
                layer_u = tf.divide(layer_y_positive - layer_b[layer_i], tf.sqrt(tf.maximum(layer_variance[layer_i], 1e-5)))
                beta = tf.divide(tf.sqrt(tf.reduce_sum(tf.multiply(layer_y, layer_y), axis=1, keep_dims=True)),
                             tf.sqrt(tf.reduce_sum(tf.multiply(layer_u, layer_u), axis=1, keep_dims=True)))
                message1 = tf.multiply(layer_u, beta)
            else:
                message1 = layer_y
            layer_messages.append(message1)

            # backward
            layer_u = layer_units[layer_i]
            if layer_i < feature_layers - 1:
                layer_u = tf.divide(layer_u, beta)
                layer_u = tf.multiply(layer_u, tf.sqrt(tf.maximum(layer_variance[layer_i], 1e-5))) + layer_b[layer_i]
                layer_y = tf.multiply(layer_u, layer_y_sign)
            else:
                layer_y = layer_u
            message2 = tf.matmul(layer_y, tf.transpose(layer_w[layer_i], [1, 0]))
            if layer_i == 0:
                input_message = message2
            else:
                layer_messages[layer_i-1] = layer_messages[layer_i-1] * 0.5 + message2 * 0.5

        label_message = input_message[:, -mnist.LABEL_PIXELS:]
        # label_message = tf.minimum(tf.maximum(label_message, 0.0), label_value)
        # label_real = label_message
        # label_prob = tf.divide(label_message, label_value)
        # rand_value = tf.random.uniform([FLAGS.batch_size, mnist.LABEL_PIXELS])
        # label_message = tf.cast(tf.greater(label_prob, rand_value), tf.float32) * label_value
        unit_op = tf.assign(label_infer, label_message)
        tf.add_to_collection("set_units", unit_op)

        for layer_i in range(feature_layers):
            unit_op = tf.assign(layer_units[layer_i], layer_messages[layer_i])
            tf.add_to_collection("set_units", unit_op)

    label_infer = tf.reshape(label_infer, [FLAGS.batch_size, mnist.PIXEL_PER_CLASS, mnist.NUM_CLASSES])
    label_infer = tf.reduce_sum(label_infer, axis=1)
    predicted_label = tf.argmax(label_infer, axis=1)

    # label_real = tf.reshape(label_real, [FLAGS.batch_size, mnist.PIXEL_PER_CLASS, mnist.NUM_CLASSES])
    # label_real = tf.reduce_sum(label_real, axis=1)
    # predicted_label = tf.argmax(label_real, axis=1)
    return activations, predicted_label


def inference(images, feature_layers):
    label_value = 1.0 / tf.sqrt(tf.constant(mnist.PIXEL_PER_CLASS, tf.float32))
    images = normalize(images)

    with tf.compat.v1.name_scope("layer_units"):
        label_infer = tf.Variable(tf.zeros([FLAGS.batch_size, mnist.LABEL_PIXELS], tf.float32), name="label_infer")
        reset_op = tf.assign(label_infer, tf.zeros([FLAGS.batch_size, mnist.LABEL_PIXELS], tf.float32))
        tf.add_to_collection("reset_label", reset_op)

    inputs = tf.concat([images, label_infer], axis=1)
    layers = 0
    activations = []
    layer_sign = []
    layer_w = []
    layer_b = []
    layer_beta = []
    layer_variance = []
    with tf.compat.v1.name_scope('hidden_layers'):
        for layer_i in range(feature_layers):
            weights = tf.Variable(
                tf.random.truncated_normal(
                    [mnist.LAYER_PIXELS, mnist.LAYER_PIXELS],
                    mean=0.0,
                    stddev=0.01
                ), name='weights')
            layer_w.append(weights)
            inputs = tf.matmul(inputs, weights)
            if layer_i < feature_layers - 1:
                mag = tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=1, keepdims=True))
                sign = tf.sign(inputs)
                layer_sign.append(sign)
                inputs = tf.pow(tf.abs(inputs), 1.0)
                input_shape = inputs.get_shape().as_list()
                b = tf.Variable(tf.zeros([input_shape[1]], tf.float32), name="norm_mean", trainable=False)
                variance = tf.Variable(tf.ones([input_shape[1]], tf.float32) * 0.1, name="norm_variance",
                                       trainable=False)
                layer_b.append(b)
                layer_variance.append(variance)
                inputs = tf.divide(inputs - b, tf.sqrt(tf.maximum(variance, 1e-5)))
                beta = tf.divide(mag, tf.maximum(
                    tf.sqrt(tf.reduce_sum(tf.multiply(inputs, inputs), axis=[1], keepdims=True)),
                    1e-5))
                layer_beta.append(beta)
                inputs = tf.multiply(inputs, beta)
            activation = tf.reduce_sum(tf.pow(inputs, 2.0), axis=1)
            activations.append(activation)
            layers = layers + 1

    with tf.compat.v1.name_scope('layer_propagation'):
        for layer_i in reversed(range(feature_layers)):
            if layer_i < feature_layers - 1:
                inputs = tf.divide(inputs, layer_beta[layer_i])
                inputs = tf.multiply(inputs, tf.sqrt(tf.maximum(layer_variance[layer_i], 1e-5))) + layer_b[layer_i]
                inputs = tf.multiply(inputs, layer_sign[layer_i])
            inputs = tf.matmul(inputs, tf.transpose(layer_w[layer_i], [1, 0]))
        label_input = inputs[:, -mnist.LABEL_PIXELS:]
        # label_input = tf.minimum(tf.maximum(label_input, 0.0), label_value)
        # label_message = tf.minimum(tf.maximum(label_message, 0.0), label_value)
        # label_real = label_message
        # label_prob = tf.divide(label_message, label_value)
        # rand_value = tf.random.uniform([FLAGS.batch_size, mnist.LABEL_PIXELS])
        # label_message = tf.cast(tf.greater(label_prob, rand_value), tf.float32) * label_value
        unit_op = tf.assign(label_infer, label_input)
        tf.add_to_collection("update_label", unit_op)

    label_square = tf.reshape(tf.multiply(label_infer, label_infer), [FLAGS.batch_size, mnist.PIXEL_PER_CLASS, mnist.NUM_CLASSES])
    label_square = tf.reduce_sum(label_square, axis=1)
    predicted_label = tf.argmax(label_square, axis=1)
    return activations, predicted_label


def run_inference():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, input_labels_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        activations, predicted_label = inference(images_placeholder, FLAGS.feature_layers)

        activation = activations[-1]
        reset_op = tf.group(tf.get_collection("unset_label"))
        update_op = tf.group(tf.get_collection("update_label"))

        # Add the variable initializer Op.
        init = tf.compat.v1.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        variables = [v for v in tf.global_variables() if v.name.find("weights") >= 0 or v.name.find("norm") >= 0]
        saver = tf.compat.v1.train.Saver(variables)

        # Create a session for running Ops on the Graph.
        sess = tf.compat.v1.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

        data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data,
                                              validation_size=FLAGS.validation_size)

        test_accuracy = do_eval(sess, predicted_label, reset_op, update_op, images_placeholder,
                              labels_placeholder, data_sets.test)

        # train_accuracy = do_eval(sess, activation, label_variable, images_placeholder,
        #                         labels_placeholder, data_sets.train)

        print("train_error %.04f test_error %.04f" % (1.0, 1.0 - test_accuracy))
        return {"training": 1.0, "test": 1.0 - test_accuracy}


def set_environ():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if len(memory_gpu) < 1:
        os.system('rm tmp')
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
    os.system('rm tmp')
    print(memory_gpu, str(np.argmax(memory_gpu)))


def main(_):
    set_environ()
    run_inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature_layers',
        type=int,
        default=1,
        help='Number of hidden layers.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--validation_size',
        type=int,
        default=0,
        help='Label size.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

