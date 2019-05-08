# 2018.08
# Input Image Format: RGB
# Input Image Range: (0, 255)
# Original Model From: Caffe

import tensorflow as tf
import numpy as np

class VGG19(object):
    def __init__(self, pretrained='vgg19.npy', keep_prob = 0.5 ,reuse=False, name='VGG19'):
        super(VGG19, self).__init__()
        self.data = np.load(pretrained).item()
        self.keep_prob = keep_prob
        self.reuse = reuse
        self.name = name
        self.layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                       'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                       'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                       'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
                       'flatten', 'fc6', 'relu1', 'dropout1', 'fc7', 'relu2', 'dropout2', 'fc8')
        self.mean = [103.939, 116.779, 123.68]
        self.net = dict()

    def conv_layer(self, input, data, name):
        w = tf.get_variable(name=name + '_W', dtype='float32', initializer=tf.constant(data[name][0]))
        b = tf.get_variable(name=name + '_b', dtype='float32', initializer=tf.constant(data[name][1]))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME', name=name)
        return tf.nn.bias_add(conv, b, name='add' + name)

    def pooling_layer(self, input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def relu_layer(self, input, name):
        return tf.nn.relu(input, name=name)

    def fc_layer(self, input, data, name):
        w = tf.get_variable(name=name + '_W', dtype='float32', initializer=tf.constant(data[name][0]))
        b = tf.get_variable(name=name + '_b', dtype='float32', initializer=tf.constant(data[name][1]))
        return tf.nn.bias_add(tf.matmul(input, w), b)

    def dropout_layer(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob=keep_prob, name=name)

    def flatten_layer(self, input, name):
        return tf.reshape(input, [-1, int(np.prod(input.get_shape()[1:]))], name=name)

    def __call__(self, input):
        mean = tf.constant(self.mean, dtype=tf.float32, shape=[1, 1, 1, 3])
        r, g, b = tf.split(input, 3, 3)
        current = tf.concat([b, g, r], 3) - mean # RGB->BGR
        # Note: This model accept BGR for real input
        with tf.variable_scope(self.name):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            for name in self.layers:
                if name.startswith('conv'):
                    current = self.conv_layer(current, self.data, name)
                elif name.startswith('relu'):
                    current = self.relu_layer(current, name)
                elif name.startswith('pool'):
                    current = self.pooling_layer(current, name)
                elif name.startswith('flatten'):
                    current = self.flatten_layer(current, name)
                elif name.startswith('fc'):
                    current = self.fc_layer(current, self.data, name)
                elif name.startswith('dropout'):
                    current = self.dropout_layer(current, self.keep_prob, name)
                self.net[name] = current
            return current

if __name__ == '__main__':

    data = np.random.randn(1, 224, 224, 3) * 25
    img = tf.placeholder(tf.float32, [None, 224, 224, 3])
    p = tf.placeholder(tf.float32, [])
    net = VGG19('vgg19.npy',keep_prob = p,reuse=False, name='vgg19')
    out = net(img)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prob = sess.run(out, feed_dict={img: data, p: 1.0})
        print(prob.shape)