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
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import mnist_bp as mnist

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
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set, simple_run):
    """Runs one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    num_examples = data_set.num_examples
    if simple_run:
        if num_examples > 10000:
            num_examples = 10000
    steps_per_epoch = num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                       FLAGS.fake_data, shuffle=False)
        true_count += sess.run(eval_correct, feed_dict={
            images_placeholder: images_feed,
            labels_placeholder: labels_feed
        })
    precision = float(true_count) / num_examples
    return precision


def run_training():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, input_labels_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        is_training = tf.Variable(tf.constant(True, tf.bool), trainable=False)
        train_outputs = mnist.inference(images_placeholder, FLAGS.feature_layers, is_training=is_training)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        loss = mnist.loss(train_outputs, labels_placeholder)

        eval_correct = mnist.evaluation(train_outputs, labels_placeholder)

        sgd_opts = mnist.training_sgd(loss, global_step, 0.01)

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

        total_steps = 100000
        total_steps = int(total_steps * np.sqrt(training_samples / 60000.0))
        total_epoches = total_steps // batch_per_epoch
        change_rate_epoches = [int(total_epoches * 0.9)]

        best_train_accuracy = 0.0
        best_validation_accuracy = 0.0
        best_test_accuracy = 0.0

        for epoch in xrange(total_epoches):
            data_sets.train.reset()
            for step in xrange(batch_per_epoch):
                images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)
                sess.run(sgd_opts, feed_dict={
                    images_placeholder: images_feed,
                    labels_placeholder: labels_feed
                })

            # if (epoch + 1) % 10 == 0:
            if (epoch + 1) % (total_epoches // 50) == 0:
                sess.run(tf.assign(is_training, tf.constant(False, tf.bool)))
                train_accuracy = do_eval(sess, eval_correct, images_placeholder,
                                         labels_placeholder, data_sets.train, True)
                test_accuracy = do_eval(sess, eval_correct, images_placeholder,
                                        labels_placeholder, data_sets.test, True)
                print("%d, %.04f, %.04f" % (epoch + 1, train_accuracy, test_accuracy))
                sess.run(tf.assign(is_training, tf.constant(True, tf.bool)))

        best_train_accuracy = do_eval(sess, eval_correct, images_placeholder,
                                      labels_placeholder, data_sets.train, False)
        best_test_accuracy = do_eval(sess, eval_correct, images_placeholder,
                                     labels_placeholder, data_sets.test, False)
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=epoch)
        print("train_accuracy %.04f, test_accuracy %.04f" % (best_train_accuracy, best_test_accuracy))


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
