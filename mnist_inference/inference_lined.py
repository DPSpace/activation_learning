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


def create_single_line(line_width):
    img = np.zeros([28, 28], np.int)
    a = np.random.randint(0, 28)
    b = np.random.randint(0, 28)
    if line_width == 3:
        if np.random.randint(0, 2) == 0:
            for i in range(28):
                x = i / 27.0 * (b - a) + a
                x1 = int(np.ceil(x))
                x2 = int(np.floor(x))
                x3 = x1 + 1
                img[x1, i] = 1
                img[x2, i] = 1
                if x3 <= 27:
                    img[x3, i] = 1
        else:
            for i in range(28):
                x = i / 27.0 * (b - a) + a
                x1 = int(np.ceil(x))
                x2 = int(np.floor(x))
                x3 = x1 + 1
                img[i, x1] = 1
                img[i, x2] = 1
                if x3 <= 27:
                    img[i, x3] = 1
    elif line_width == 2:
        if np.random.randint(0, 2) == 0:
            for i in range(28):
                x = i / 27.0 * (b - a) + a
                x1 = int(np.ceil(x))
                x2 = int(np.floor(x))
                img[x1, i] = 1
                img[x2, i] = 1
        else:
            for i in range(28):
                x = i / 27.0 * (b - a) + a
                x1 = int(np.ceil(x))
                x2 = int(np.floor(x))
                img[i, x1] = 1
                img[i, x2] = 1
    elif line_width == 1:
        if np.random.randint(0, 2) == 0:
            for i in range(28):
                x = i / 27.0 * (b - a) + a
                x1 = int(np.ceil(x))
                x2 = int(np.floor(x))
                if x1 - x < x - x2:
                    img[x1, i] = 1
                else:
                    img[x2, i] = 1
        else:
            for i in range(28):
                x = i / 27.0 * (b - a) + a
                x1 = int(np.ceil(x))
                x2 = int(np.floor(x))
                if x1 - x < x - x2:
                    img[i, x1] = 1
                else:
                    img[i, x2] = 1
    return np.reshape(img, [28 * 28])


def line_images(images_feed, line_width):
    for i in range(np.shape(images_feed)[0]):
        img = images_feed[i]
        line = create_single_line(line_width)
        img = img + line
        img = np.greater(img, 0.5)
        images_feed[i] = img
    return images_feed


def one_hot(labels_feed):
    label_vector = np.eye(10)[labels_feed]
    label_vector = np.divide(label_vector, np.sqrt(mnist.PIXEL_PER_CLASS))
    # return np.pad(label_vector, ((0, 0), (0, (mnist.PIXEL_PER_CLASS - 1) * mnist.NUM_CLASSES)),  "constant")
    return np.tile(label_vector, [1, mnist.PIXEL_PER_CLASS])


def do_eval(sess,
            activation,
            input_labels_placeholder,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    print(steps_per_epoch)
    if steps_per_epoch > 100:
        steps_per_epoch = 100
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        if step % 10 == 0:
            print(FLAGS.line_width, step)
        images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                       FLAGS.fake_data)
        images_feed = line_images(images_feed, FLAGS.line_width)
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


def run_inference():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, input_labels_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        activations = mnist.inference(images_placeholder, input_labels_placeholder, FLAGS.feature_layers,
                                tf.ones([FLAGS.batch_size], tf.float32))
        activation = activations[-1]

        # Add the variable initializer Op.
        init = tf.compat.v1.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        # variables = [v for v in tf.global_variables() if v.name.find("weights") >= 0 or v.name.find("norm") >= 0]
        saver = tf.compat.v1.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.compat.v1.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

        data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data,
                                              validation_size=FLAGS.validation_size)

        test_accuracy = do_eval(sess, activation, input_labels_placeholder, images_placeholder,
                               labels_placeholder, data_sets.test)
        # train_accuracy = do_eval(sess, activation, , images_placeholder,
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


def run_experiments6():
    path = os.path.join("data", "experiments6_nofeedback.txt")
    if os.path.exists(path):
        os.remove(path)

    for i in range(1, 4):
        FLAGS.line_width = i
        FLAGS.validation_size = 0
        res = run_inference()
        with open(path, "a", encoding="utf-8") as f:
            f.write("%d, %.04f, %.04f\n"
                    % (i, res['training'] * 100.0, res['test'] * 100.0))
            print("lines %d, training %.04f, test %.04f\n"
                    % (i, res['training'] * 100.0, res['test'] * 100.0))


def main(_):
    set_environ()
    if FLAGS.experiments6:
        return run_experiments6()
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
    parser.add_argument(
        '--experiments6',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

