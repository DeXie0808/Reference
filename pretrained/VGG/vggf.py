# 2018.08
# Input Image Format: RGB
# Input Image Range: (0, 255)
# Original Model From: Matconvnet

import tensorflow as tf
import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

class VGG_F(object):
    def __init__(self, pretrained='vggf.npy', name='VGG_F'):
        super(VGG_F, self).__init__()

        data = np.load(pretrained).item()
        self.name = name
        self.layers = ('conv1', 'relu1', 'norm1', 'pool1','conv2', 'relu2', 'norm2', 'pool2',
                       'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5', 'pool5', 'fc6',
                       'relu6', 'fc7', 'relu7','fc8')
        self.weights = data['layers'][0]
        self.mean = data['normalization'][0][0][0]
        self.net = dict()

    def conv_layer(self, input, weights, bias, pad, stride, i):
        pad = pad[0]
        stride = stride[0]
        input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
        w = tf.get_variable(name='w' + str(i), dtype='float32', initializer=tf.constant(weights))
        b = tf.get_variable(name='bias' + str(i), dtype='float32', initializer=tf.constant(bias))
        conv = tf.nn.conv2d(input, w, strides=[1, stride[0], stride[1], 1], padding='VALID', name='conv' + str(i))
        return tf.nn.bias_add(conv, b, name='add' + str(i))

    def full_conv(self, input, weights, bias, i):
        w = tf.get_variable(name='w' + str(i), dtype='float32', initializer=tf.constant(weights))
        b = tf.get_variable(name='bias' + str(i), dtype='float32', initializer=tf.constant(bias))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID', name='fc' + str(i))
        return tf.nn.bias_add(conv, b, name='add' + str(i))

    def pool_layer(self, input, stride, pad, area):
        pad = pad[0]
        area = area[0]
        stride = stride[0]
        input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
        return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1],
                              padding='VALID')

    def __call__(self, input):

        current = tf.convert_to_tensor((input - tf.constant(self.mean, shape=[1,224,224,3], dtype=tf.float32)), dtype=tf.float32)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            for i, name in enumerate(self.layers[:-1]):
                if name.startswith('conv'):
                    kernels, bias = self.weights[i][0][0][0][0]
                    bias = bias.reshape(-1)
                    pad = self.weights[i][0][0][1]
                    stride = self.weights[i][0][0][4]
                    current = self.conv_layer(current, kernels, bias, pad, stride, i)
                elif name.startswith('relu'):
                    current = tf.nn.relu(current)
                elif name.startswith('pool'):
                    stride = self.weights[i][0][0][1]
                    pad = self.weights[i][0][0][2]
                    area = self.weights[i][0][0][5]
                    current = self.pool_layer(current, stride, pad, area)
                elif name.startswith('fc'):
                    kernels, bias = self.weights[i][0][0][0][0]
                    bias = bias.reshape(-1)
                    current = self.full_conv(current, kernels, bias, i)
                elif name.startswith('norm'):
                    current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001,
                                                                 beta=0.75)
                self.net[name] = current
            return current



if __name__ == '__main__':

    data = np.random.randn(1,224,224,3) * 255.

    img = tf.placeholder(dtype=tf.float32, shape=[None, 224,224, 3], name='input')
    net = VGG_F(pretrained='/home/xd/my_project/Conference/NIPS2019/for_lichao/DCMH_PRO/refernce_code/pretrained/VGG/vggf.npy', name='CNNF')
    out = net(img)
    with tf.Session() as sess :
        tf.global_variables_initializer().run()
        # a = tf.trainable_variables()
        # for i in a:
        #     print i.name
        b = sess.run(out, feed_dict={img:data})
        print(b.shape)
