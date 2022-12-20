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


def do_eval(sess,
            activation,
            images_placeholder,
            input_labels_placeholder,
            labels_placeholder,
            data_set, simple_run):
    """Runs one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    local_batch_size = FLAGS.batch_size
    steps_per_epoch = data_set.num_examples // local_batch_size
    if simple_run:
        if steps_per_epoch > 2000 // local_batch_size:
            steps_per_epoch = 2000 // local_batch_size
    if steps_per_epoch > 10000 // local_batch_size:
        steps_per_epoch = 10000 // local_batch_size
    num_examples = steps_per_epoch * local_batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = data_set.next_batch(local_batch_size,
                                                       FLAGS.fake_data, shuffle=False)
        outputs = []
        for i in range(10):
            feed_dict = {
                images_placeholder: images_feed,
                input_labels_placeholder: labels_feed - labels_feed + i,
                labels_placeholder: labels_feed
            }
            outputs.append(sess.run(activation, feed_dict=feed_dict))
        result = np.argmax(np.array(outputs), axis=0)
        true_count += np.sum(np.equal(result, labels_feed))
    precision = float(true_count) / num_examples
    return precision


def do_activation_value(sess,
            activation,
            images_placeholder,
            input_labels_placeholder,
            labels_placeholder,
            data_set, simple_run):
    sum_activation = 0  # Counts the number of correct predictions.
    local_batch_size = FLAGS.batch_size
    steps_per_epoch = data_set.num_examples // local_batch_size
    if simple_run:
        if steps_per_epoch > 10000 // local_batch_size:
            steps_per_epoch = 10000 // local_batch_size
    num_examples = steps_per_epoch * local_batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = data_set.next_batch(local_batch_size,
                                                       FLAGS.fake_data, shuffle=False)
        feed_dict = {
            images_placeholder: images_feed,
            input_labels_placeholder: labels_feed,
            labels_placeholder: labels_feed
        }
        sum_activation += sess.run(tf.reduce_sum(activation, axis=0), feed_dict=feed_dict)
    mean_activation = float(sum_activation) / num_examples
    return mean_activation


def run_training():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, input_labels_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)
        learning_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size))

        # Build a Graph that computes predictions from the inference model.
        activations = mnist.inference(images_placeholder, input_labels_placeholder,
                                                     FLAGS.feature_layers, learning_placeholder)
        activation = activations[-1]
        train_ops = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

        # Add the variable initializer Op.
        init = tf.compat.v1.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.compat.v1.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.compat.v1.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data,
                                              validation_size=60000-FLAGS.training_size)

        training_samples = data_sets.train.num_examples
        batch_per_epoch = training_samples // FLAGS.batch_size
        total_steps = 600 * 50
        total_epoches = total_steps // batch_per_epoch
        change_rate_epoches = []
        learning_rate = FLAGS.learning_rate

        one_rate = np.ones([FLAGS.batch_size], np.float)
        moving_factor = 1.0

        for epoch in xrange(total_epoches):
            if epoch in change_rate_epoches:
                learning_rate = learning_rate * 0.1

            data_sets.train.reset()
            for step in xrange(batch_per_epoch):

                if FLAGS.feedback:
                    images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
                    feed_dict = {
                        images_placeholder: images_feed,
                        input_labels_placeholder: labels_feed,
                        labels_placeholder: labels_feed,
                        learning_placeholder: one_rate
                    }
                    true_value = sess.run(activation, feed_dict=feed_dict)

                    outputs = []
                    for i in range(1, 10):
                        try_label = (labels_feed + i) % 10
                        outputs.append(sess.run(activation, feed_dict={
                            images_placeholder: images_feed,
                            input_labels_placeholder: try_label,
                            labels_placeholder: labels_feed,
                            learning_placeholder: one_rate
                        }))
                    outputs = np.array(outputs)
                    result = np.argmax(outputs, axis=0)
                    try_label = (labels_feed + 1 + result) % 10
                    try_value = np.max(outputs, axis=0)

                    unlearn_rate = np.minimum(np.maximum((try_value + 0.2 - true_value) * 5.0, 0.00), 1.0) \
                        if training_samples > 1000 else np.minimum(
                        np.maximum((try_value + 0.5 - true_value) * 2.0, 0.00), 1.0)

                    decay = 0.01
                    moving_factor = moving_factor * (1.0 - decay) + np.mean(unlearn_rate) * decay
                    unlearn_rate = np.divide(unlearn_rate, np.maximum(moving_factor, 1e-5))

                    sess.run(train_ops, feed_dict={
                        images_placeholder: images_feed,
                        input_labels_placeholder: labels_feed,
                        labels_placeholder: labels_feed,
                        learning_placeholder: unlearn_rate * learning_rate
                    })

                    sess.run(train_ops, feed_dict={
                        images_placeholder: images_feed,
                        input_labels_placeholder: try_label,
                        labels_placeholder: try_label,
                        learning_placeholder: - unlearn_rate * 0.9 * learning_rate
                    })

                else:
                    images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)

                    sess.run(train_ops, feed_dict={
                        images_placeholder: images_feed,
                        input_labels_placeholder: labels_feed,
                        labels_placeholder: labels_feed,
                        learning_placeholder: one_rate * learning_rate
                    })

            if (epoch +1) % (total_epoches // 500) == 0:
                # train_accuracy = do_eval(sess, activation, images_placeholder, input_labels_placeholder,
                #                          labels_placeholder, data_sets.train, False)
                test_accuracy = do_eval(sess, activation, images_placeholder, input_labels_placeholder,
                                        labels_placeholder, data_sets.test, False)
                # mean_activation = do_activation_value(sess, activation, images_placeholder,
                #                                       input_labels_placeholder,
                #                                       labels_placeholder, data_sets.train, True)
                # print("%d, %.04f, %.04f, %.04f" % (epoch + 1, train_accuracy, test_accuracy, mean_activation))
                print("%d, %.04f" % (epoch + 1, test_accuracy))


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
    if tf.io.gfile.exists(FLAGS.log_dir):
        tf.io.gfile.rmtree(FLAGS.log_dir)
    tf.io.gfile.makedirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--feature_layers',
        type=int,
        default=1,
        help='Number of feature layers.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--training_size',
        type=int,
        default=55000,
        help='Size of the validation set.'
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

    parser.add_argument(
        '--feedback',
        default=False,
        help='If true, uses feedback to unlearn negative samples.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
