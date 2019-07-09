from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import glob
# from skimage import io, transform
from tensorflow.python.framework import graph_util
import collections

# path = 'vgg16/picture/'
w = 224
h = 224
c = 1


# def build_network(height, width, channel):
    # x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
    # y = tf.placeholder(tf.float32, shape=[None, 1], name='labels_placeholder')




    # return dict(
        # x=x,
        # y=y,
        # optimize=optimize,
        # cost=cost,
        # # correct_prediction=correct_prediction,
        # # correct_times_in_batch=correct_times_in_batch,
    # )

    

    
class Model:
    def __init__(self, data_X, data_y_x, data_y_y):
        # self.n_class = 10
        self._create_architecture(data_X, data_y_x, data_y_y)

    def _create_architecture(self, data_X, data_y_x, data_y_y):
        # y_hot = tf.one_hot(data_y, depth = self.n_class)
        x_pred, y_pred = self._create_model(data_X)
        # predictions = tf.argmax(logits, 1, output_type = tf.int32)
        # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_hot, 
                                                                              # logits = logits))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        # self.accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, data_y), tf.float32))
        import pdb
        # pdb.set_trace()
        self.cost = tf.reduce_sum(tf.pow(data_y_x - x_pred, 2))/2
        # self.cost = tf.reduce_mean(tf.losses.mean_squared_error(data_y_x, x_pred))
        self.optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)

    def _create_model(self, x):
        # conv1
        with tf.name_scope('conv1_1') as scope:
            kernel = self.weight_variable([3, 3, 1, 32])
            biases = self.bias_variable([32])
            output_conv1_1 = tf.nn.relu(self.conv2d(x, kernel) + biases, name=scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = self.weight_variable([3, 3, 32, 32])
            biases = self.bias_variable([32])
            output_conv1_2 = tf.nn.relu(self.conv2d(output_conv1_1, kernel) + biases, name=scope)

        pool1 = self.pool_max(output_conv1_2)

        # conv2
        with tf.name_scope('conv2_1') as scope:
            kernel = self.weight_variable([3, 3, 32, 64])
            biases = self.bias_variable([64])
            output_conv2_1 = tf.nn.relu(self.conv2d(pool1, kernel) + biases, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = self.weight_variable([3, 3, 64, 64])
            biases = self.bias_variable([64])
            output_conv2_2 = tf.nn.relu(self.conv2d(output_conv2_1, kernel) + biases, name=scope)

        pool2 = self.pool_max(output_conv2_2)

        # conv3
        with tf.name_scope('conv3_1') as scope:
            kernel = self.weight_variable([3, 3, 64, 128])
            biases = self.bias_variable([128])
            output_conv3_1 = tf.nn.relu(self.conv2d(pool2, kernel) + biases, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = self.weight_variable([3, 3, 128, 128])
            biases = self.bias_variable([128])
            output_conv3_2 = tf.nn.relu(self.conv2d(output_conv3_1, kernel) + biases, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = self.weight_variable([3, 3, 128, 128])
            biases = self.bias_variable([128])
            output_conv3_3 = tf.nn.relu(self.conv2d(output_conv3_2, kernel) + biases, name=scope)

        pool3 = self.pool_max(output_conv3_3)

        # conv4
        with tf.name_scope('conv4_1') as scope:
            kernel = self.weight_variable([3, 3, 128, 256])
            biases = self.bias_variable([256])
            output_conv4_1 = tf.nn.relu(self.conv2d(pool3, kernel) + biases, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            biases = self.bias_variable([256])
            output_conv4_2 = tf.nn.relu(self.conv2d(output_conv4_1, kernel) + biases, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            biases = self.bias_variable([256])
            output_conv4_3 = tf.nn.relu(self.conv2d(output_conv4_2, kernel) + biases, name=scope)

        pool4 = self.pool_max(output_conv4_3)

        # conv5
        with tf.name_scope('conv5_1') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            biases = self.bias_variable([256])
            output_conv5_1 = tf.nn.relu(self.conv2d(pool4, kernel) + biases, name=scope)

        with tf.name_scope('conv5_2') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            biases = self.bias_variable([256])
            output_conv5_2 = tf.nn.relu(self.conv2d(output_conv5_1, kernel) + biases, name=scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            biases = self.bias_variable([256])
            output_conv5_3 = tf.nn.relu(self.conv2d(output_conv5_2, kernel) + biases, name=scope)

        pool5 = self.pool_max(output_conv5_3)

        #fc6
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            kernel = self.weight_variable([shape, 4096])
            biases = self.bias_variable([4096])
            pool5_flat = tf.reshape(pool5, [-1, shape])
            output_fc6 = tf.nn.relu(self.fc(pool5_flat, kernel, biases), name=scope)

        #fc7
        with tf.name_scope('fc7') as scope:
            kernel = self.weight_variable([4096, 4096])
            biases = self.bias_variable([4096])
            output_fc7 = tf.nn.relu(self.fc(output_fc6, kernel, biases), name=scope)

        #fc8
        with tf.name_scope('fc8') as scope:
            kernel = self.weight_variable([4096, 3])
            biases = self.bias_variable([3])
            output_fc8 = tf.nn.relu(self.fc(output_fc7, kernel, biases), name=scope)

        x_pred = tf.nn.sigmoid(output_fc8[0], name='x_pred')
        y_pred = tf.nn.sigmoid(output_fc8[1], name='y_pred')

        return x_pred, y_pred
    
    
    def weight_variable(self, shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

    def pool_max(self, input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def fc(self, input, w, b):
        return tf.matmul(input, w) + b
