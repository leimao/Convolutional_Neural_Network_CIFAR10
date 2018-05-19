import tensorflow as tf
import os

class CNN(object):

    def __init__(self, input_size, num_classes, optimizer):

        self.num_classes = num_classes
        self.input_size = input_size
        self.optimizer = optimizer

        self.learning_rate = tf.placeholder(tf.float32, shape = [], name = 'learning_rate')
        self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')
        self.input = tf.placeholder(tf.float32, [None] + self.input_size, name = 'input')
        self.label = tf.placeholder(tf.float32, [None, self.num_classes], name = 'label')
        self.output = self.network_initializer()
        self.loss = self.loss_initializer()
        self.optimization = self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def network(self, input, dropout_rate):

        conv1 = tf.layers.conv2d(
            inputs = input,
            filters = 64,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv1')

        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 64,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv2')

        pool1 = tf.layers.max_pooling2d(
            inputs = conv2,
            pool_size = [2, 2],
            strides = [2, 2],
            name = 'pool1')

        pool1_dropout = tf.layers.dropout(
            inputs = pool1,
            rate = dropout_rate,
            name = 'pool1_dropout')

        conv3 = tf.layers.conv2d(
            inputs = pool1_dropout,
            filters = 128,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv3')

        conv4 = tf.layers.conv2d(
            inputs = conv3,
            filters = 128,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv4')

        pool2 = tf.layers.max_pooling2d(
            inputs = conv4,
            pool_size = [2, 2],
            strides = [2, 2],
            name = 'pool2')

        pool2_dropout = tf.layers.dropout(
            inputs = pool2,
            rate = dropout_rate,
            name = 'pool2_dropout')

        conv5 = tf.layers.conv2d(
            inputs = pool2_dropout,
            filters = 256,
            kernel_size = [3, 3],
            padding = 'same',
            activation = tf.nn.relu,
            name = 'conv5')

        pool3 = tf.layers.max_pooling2d(
            inputs = conv5,
            pool_size = [2, 2],
            strides = [2, 2],
            name = 'pool3')

        pool3_dropout = tf.layers.dropout(
            inputs = pool3,
            rate = dropout_rate,
            name = 'pool3_dropout')

        flat = tf.layers.flatten(
            inputs = pool3_dropout, 
            name = 'flat')

        fc1 = tf.layers.dense(
            inputs = flat,
            units = 256,
            activation = tf.nn.relu,
            name = 'fc1')

        fc1_dropout = tf.layers.dropout(
            inputs = fc1,
            rate = dropout_rate,
            name = 'fc1_dropout')

        output = tf.layers.dense(
            inputs = fc1_dropout,
            units = self.num_classes,
            activation = None,
            name = 'fc2')

        return output

    def network_initializer(self):

        with tf.variable_scope('convolution_network') as scope:
            ouput = self.network(input = self.input, dropout_rate = self.dropout_rate)

        return ouput

    def loss_initializer(self):

        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = self.label, logits = self.output, name = 'cross_entropy')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy_mean')
        return cross_entropy_mean

    def optimizer_initializer(self):

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

        return optimizer

    def train(self, data, label, learning_rate, dropout_rate):

        _, train_loss = self.sess.run([self.optimization, self.loss], 
            feed_dict = {self.input: data, self.label: label, self.learning_rate: learning_rate, self.dropout_rate: dropout_rate})
        return train_loss

    def validate(self, data, label):

        output, validate_loss = self.sess.run([self.output, self.loss], 
            feed_dict = {self.input: data, self.label: label, self.dropout_rate: 0})
        return output, validate_loss

    def test(self, data):

        output = self.sess.run(self.output, feed_dict = {self.input: data, self.dropout_rate: 0})

        return output

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)

