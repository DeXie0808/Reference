import numpy as np
import os
import tensorflow as tf

from tensorflow.contrib.slim import nets
import resnet

slim = tf.contrib.slim


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    resnet_model_path = 'resnet_v1_152.ckpt'

    num_classes = 5
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='inputs')
    is_training = tf.placeholder(tf.bool, name='is_training')

    with slim.arg_scope(resnet.resnet_arg_scope()):
        net, endpoints = resnet.resnet_v1_152(inputs, num_classes=None,
                                                     is_training=is_training)

    with tf.variable_scope('Logits'):
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        net = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='fc')

    checkpoint_exclude_scopes = 'Logits'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []


    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)



    init = tf.global_variables_initializer()

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        sess.run(init)

        # Load the pretrained checkpoint file xxx.ckpt
        saver_restore.restore(sess, resnet_model_path)

        logits = sess.run(net, feed_dict={inputs: np.random.random([3, 224, 224, 3]), is_training: False})

        print(logits)
        print(logits.shape)


