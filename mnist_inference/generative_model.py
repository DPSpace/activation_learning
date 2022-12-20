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
from PIL import Image
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import mnist_activation as mnist

# Basic model parameters as external flags.
FLAGS = None

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


def inference(images, labels, feature_layers, learning_factor):
    # labels = tf.divide(labels, tf.sqrt(tf.reduce_sum(tf.multiply(labels, labels), axis=1, keep_dims=True)))
    # inputs = tf.concat([images, labels], axis=1)
    inputs = mnist.normalized(images, labels)
    layers = 0
    activation = 0
    with tf.compat.v1.name_scope('hidden_layers'):
        for layer_i in range(feature_layers):
            weights = tf.Variable(
                tf.random.truncated_normal(
                    [mnist.LAYER_PIXELS, mnist.LAYER_PIXELS],
                    mean=0.0,
                    stddev=0.01
                ), name='weights')
            inputs = mnist.fc_layer(inputs, weights, learning_factor)
            if layer_i < feature_layers - 1:
                inputs = mnist.nonlinear_fun(inputs)
            activation = tf.pow(inputs, 2.0)
            layers = layers + 1
    if FLAGS.add_noise:
        activation_scale = 1.0 + tf.constant(np.random.uniform(-0.03, 0.03, [FLAGS.batch_size, mnist.LAYER_PIXELS]), tf.float32)
        activation = tf.multiply(activation, activation_scale)
    activation = tf.reduce_sum(activation, axis=1)
    return activation


def concat_img(imgs, feature_dim):
    img_list = []
    for i in range(feature_dim[0]):
        img = np.concatenate(imgs[i * feature_dim[1]: (i+1) * feature_dim[1]], axis=1)
        img_list.append(img)
    img = np.concatenate(img_list, axis=0)
    return img


def border_img(img):
    img[0,:] = 1.0
    img[-1,:] = 1.0
    img[:,0] = 1.0
    img[:,-1] = 1.0
    return img


def print_images(images, filename):
    images = np.divide(images, np.max(images, axis=1, keepdims=True))
    images = np.minimum(np.multiply(images, np.greater(images, 0.25)) * 1.1, 1.0)
    imgs = []
    for i in range(100):
        img = np.tile(np.reshape(images[i], [28, 28, 1]), [1, 1, 3])
        img = (border_img(img) * 255).astype('uint8')
        imgs.append(img)
    imgs = concat_img(imgs, [10, 10])
    img = Image.fromarray(imgs, mode='RGB')
    img.save("data/%s.png" % filename)
    # img.save("data/%s.eps" % filename)


def run_inference():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Build a Graph that computes predictions from the inference model.
        init_image = tf.random.uniform([FLAGS.batch_size, mnist.IMAGE_PIXELS],
                                       minval=0.0,
                                       maxval=1.0,
                                       dtype=tf.float32)
        inputs_mag = tf.sqrt(tf.reduce_sum(tf.multiply(init_image, init_image), axis=1, keep_dims=True))
        init_image = tf.divide(init_image, tf.maximum(inputs_mag, 10e-8))
        image_variable = tf.Variable(init_image, name="image_variable")
        label_value = np.reshape(np.tile(np.reshape(np.arange(10), [10, 1]), [1, 10]), [100])
        labels = tf.constant(label_value, tf.int32)
        labels = tf.one_hot(labels, depth=NUM_CLASSES, on_value=1.0, off_value=0.0, axis=-1)
        labels = tf.divide(labels, tf.sqrt(tf.constant(PIXEL_PER_CLASS, tf.float32)))
        labels = tf.tile(labels, [1, PIXEL_PER_CLASS])

        activation = inference(image_variable, labels, FLAGS.feature_layers, tf.ones([FLAGS.batch_size], tf.float32))
        global_step = tf.Variable(0, name='global_step', trainable=False)

        norm_image = tf.divide(image_variable, tf.sqrt(tf.reduce_sum(tf.multiply(image_variable, image_variable), axis=1, keepdims=True)))
        # Gradient
        mag = tf.sqrt(tf.reduce_sum(tf.multiply(image_variable, image_variable), axis=1))
        target_function = tf.multiply(mag - 1.0, mag - 1.0) * 0.1 \
                          + tf.reduce_sum(tf.abs(norm_image), axis=1) * 0.003 \
                          - tf.pow(activation, 1.0)

        learning_rate = tf.Variable(tf.constant(0.5, tf.float32), trainable=False, name="learning_rate")
        rate_op = tf.assign(learning_rate, learning_rate * 0.5)
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
        train_op = optimizer.minimize(target_function, global_step=global_step, var_list=[image_variable])
        norm_op = tf.assign(image_variable, tf.divide(image_variable,
                    tf.sqrt(tf.reduce_sum(tf.multiply(image_variable, image_variable), axis=1, keep_dims=True))))
        # gradient = tf.gradients(target_function, image_variable)[0]
        # gradient = tf.divide(gradient, tf.sqrt(tf.reduce_sum(tf.multiply(gradient, gradient), axis=1, keep_dims=True)))
        # train_op = tf.assign(image_variable, image_variable - tf.multiply(gradient, learning_rate))

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

        image_file_name = "generated_images" + "_noise" if FLAGS.add_noise else "_org"

        for i in range(3000000):
            # if i == 6000000 or i == 8000000:
            #    tf.assign(learning_rate, tf.multiply(learning_rate, 0.1))
            if i % 10000 == 0:
                print("step %d mag %f" % (i, sess.run(tf.reduce_mean(tf.reduce_sum(tf.multiply(image_variable, image_variable), axis=1)))))
                if i % 100000 == 0:
                    images = sess.run(image_variable)
                    print_images(images, image_file_name)
                    sess.run(rate_op)
            sess.run([train_op])
            # if i % 10 == 0:
            #     sess.run([norm_op])
        print("end")

        images = sess.run(image_variable)
        print_images(images, image_file_name)


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
    parser.add_argument(
        '--add_noise',
        default=False,
        help='If true, add noise to the output layer.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

