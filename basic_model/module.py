from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
import tensorflow as tf
import tensorflow.contrib.slim as slim


def flatten_dense(inputs,
                  num_outputs,
                  activation_fn=tf.nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=slim.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=tf.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
    with tf.variable_scope(scope, 'flatten_dense', [inputs]):
        if inputs.shape.ndims > 2:
            inputs = slim.flatten(inputs)
        return slim.fully_connected(inputs,
                                    num_outputs,
                                    activation_fn,
                                    normalizer_fn,
                                    normalizer_params,
                                    weights_initializer,
                                    weights_regularizer,
                                    biases_initializer,
                                    biases_regularizer,
                                    reuse,
                                    variables_collections,
                                    outputs_collections,
                                    trainable,
                                    scope)
def leaky_relu(x, leak=0.5):
    return tf.maximum(x, leak*x)

def i_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset



conv = partial(slim.conv2d, activation_fn=None, weights_initializer=slim.xavier_initializer(), biases_initializer=tf.zeros_initializer())
dconv = partial(slim.conv2d_transpose, activation_fn=None)
flatten = slim.flatten
fc = partial(flatten_dense, activation_fn=None)
relu = tf.nn.relu
lrelu = leaky_relu
tanh = tf.nn.tanh
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = i_norm





