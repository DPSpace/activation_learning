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

import tensorflow as tf
import numpy as np
import cifar10
import cifar10_input
# import cifar10_input_noaug as cifar10_input
import cifar10_inference as cifar
import time

import argparse
import os
import sys
import random
from six.moves import xrange  # pylint: disable=redefined-builtin

# Basic model parameters as external flags.
FLAGS = None


def do_eval(sess,
            activation,
            images_placeholder,
            input_labels_placeholder,
            labels_placeholder,
            images, labels, isEvaluate, simple_run):
    """Runs one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    num_examples = 10000 if isEvaluate else FLAGS.training_size
    if simple_run:
        if num_examples > 2000:
            num_examples = 2000
    steps_per_epoch = num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = sess.run([images, labels])
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


def run_training():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, cifar10_input.IMAGE_SIZE,
                                                                     cifar10_input.IMAGE_SIZE, 3])
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size])
        images_train, labels_train = cifar10_input.distorted_inputs(FLAGS.training_size - 5000, FLAGS.batch_size)
        images_validation, labels_validation = cifar10_input.validation_inputs(5000, FLAGS.batch_size)

        images_test, labels_test = cifar10_input.inputs(True, FLAGS.training_size, FLAGS.batch_size)

        input_labels_placeholder = tf.compat.v1.placeholder(tf.int32, shape=[FLAGS.batch_size])
        learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size])
        activations = cifar.inference(images_placeholder, input_labels_placeholder, FLAGS.feature_layers,
                                      FLAGS.kernel_size, FLAGS.channels, learning_rate_placeholder)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(update_ops)

        # Add the variable initializer Op.
        init = tf.compat.v1.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.compat.v1.train.Saver()

        # Create a session for running Ops on the Graph.
        # sess = tf.compat.v1.Session()
        sess = tf.InteractiveSession()

        # Run the Op to initialize the variables.
        sess.run(init)

        # 引入多线程
        # tf.train.start_queue_runners()
        tf.train.start_queue_runners(sess=sess)

        total_steps = 800000
        learning_value = 0.2  # 0.5 for augmentation, 0.3 for no augment
        sess.run([tf.assign(global_step, 0)])
        one_vector = np.ones([FLAGS.batch_size], np.float)
        steps_per_epoch = FLAGS.training_size // FLAGS.batch_size

        best_validation_accuracy = 0.0
        best_training_accuracy = 0.0
        best_test_accuracy = 0.0

        for step in xrange(total_steps):
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            images_feed, labels_feed = sess.run([images_train, labels_train])

            if not FLAGS.nofeedback:
                activation = activations[-1]
                outputs = []
                for i in range(1, 10):
                    try_labels = (labels_feed + i) % 10
                    activation_result = sess.run(activation, feed_dict={
                        images_placeholder: images_feed,
                        input_labels_placeholder: try_labels,
                        labels_placeholder: try_labels
                    })
                    outputs.append(activation_result)
                max_label = np.argmax(np.array(outputs), axis=0)
                deactive_label = (labels_feed + max_label + 1) % 10
                max_value = np.max(np.array(outputs), axis=0)
                true_value = sess.run(activation, feed_dict={
                    images_placeholder: images_feed,
                    input_labels_placeholder: labels_feed,
                    labels_placeholder: labels_feed
                })

                need_train = np.minimum(np.maximum((max_value + 0.2 - true_value) * 5.0, 0.0), 1.0)
                sample_rate = np.greater(true_value, 3.0 - FLAGS.feature_layers / 2.0).astype(np.float)

                sess.run(train_op, feed_dict={
                    images_placeholder: images_feed,
                    input_labels_placeholder: deactive_label,
                    labels_placeholder: deactive_label,
                    # 0.95 for augmentation, 0.7 for no augment
                    learning_rate_placeholder: -np.multiply(sample_rate * (learning_value * 0.7), need_train)
                })

                sess.run(train_op, feed_dict={
                    images_placeholder: images_feed,
                    input_labels_placeholder: labels_feed,
                    labels_placeholder: labels_feed,
                    learning_rate_placeholder: learning_value * need_train
                })
            else:
                sess.run(train_op, feed_dict={
                    images_placeholder: images_feed,
                    input_labels_placeholder: labels_feed,
                    labels_placeholder: labels_feed,
                    learning_rate_placeholder: learning_value * one_vector
                })

            if (step + 1) % 1000 == 0:
                if (step + 1) in [600000,700000]:
                    learning_value = learning_value * 0.1

                activation_list = []

                for i in range(len(activations)):
                    activation = sess.run(tf.reduce_mean(activations[i]), feed_dict={
                        images_placeholder: images_feed,
                        input_labels_placeholder: labels_feed,
                        labels_placeholder: labels_feed
                    })
                    activation_list.append(activation)

                validation_accuracy = do_eval(sess, activations[-1], images_placeholder, input_labels_placeholder,
                                             labels_placeholder, images_validation, labels_validation, False, False)

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_training_accuracy = do_eval(sess, activations[-1], images_placeholder, input_labels_placeholder,
                                         labels_placeholder, images_train, labels_train, False, False)
                    best_test_accuracy = do_eval(sess, activations[-1], images_placeholder, input_labels_placeholder,
                                        labels_placeholder, images_test, labels_test, True, False)
                    print("%d, %.04f, %0.4f" % (step + 1, best_training_accuracy, best_test_accuracy), activation_list)
                elif (step + 1) % 10000 == 0:
                    train_accuracy = do_eval(sess, activations[-1], images_placeholder, input_labels_placeholder,
                                             labels_placeholder, images_train, labels_train, False, False)
                    test_accuracy = do_eval(sess, activations[-1], images_placeholder, input_labels_placeholder,
                                                labels_placeholder, images_test, labels_test, True, False)
                    print("%d, %.04f, %0.4f" % (step+1, train_accuracy, test_accuracy), activation_list)
                else:
                    test_accuracy = do_eval(sess, activations[-1], images_placeholder, input_labels_placeholder,
                                                labels_placeholder, images_test, labels_test, True, True)
                    print("%d, %.04f, %.04f" % (step+1, test_accuracy, activation_list[-1]))

        print("train accuracy %.04f, test accuracy %0.4f" % (best_training_accuracy, best_test_accuracy))


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
        '--training_size',
        type=int,
        default=50000,
        help='Label size.'
    )
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=5,
        help='Kernel Size.'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=9,
        help='Kernel Size.'
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
        '--nofeedback',
        default=False,
        help='If true, does not use feedback.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
