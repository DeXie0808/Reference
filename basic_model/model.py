import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial
from pretrained import resnet152
from pretrained.RESNET import resnet
from pretrained import vggf
from basic_model.module import *



# def img_feature_ext(img, is_training=True):
#     with slim.arg_scope(resnet.resnet_arg_scope()):
#         feature, endpoints = resnet.resnet_v1_152(img, num_classes=None, is_training=is_training)
#     return feature


def img_feature_ext(img, pretrained, name='vggf'):
    cnn = vggf(pretrained=pretrained, name=name)
    feature = cnn(img)
    return feature


def imghash(feature, bit, name='img_hash'):
    fc_tanh = partial(fc, activation_fn=tanh)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = fc_tanh(feature, bit)
        return tf.transpose(out)


def txthash(txt, bit, name='txt_hash'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv_relu = partial(conv, activation_fn=relu)
        fc_tanh = partial(fc, activation_fn=tanh)
        out = conv_relu(txt, 8192, 1, 1)
        out = fc_tanh(out, bit)
        return tf.transpose(out)