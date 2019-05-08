from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

def tensors_filter(tensors, filters, combine_type='or'):
    assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
    assert isinstance(filters, (str, list, tuple)), '`filters` should be a string or a list(tuple) of strings!'
    assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

    if isinstance(filters, str):
        filters = [filters]

    f_tens = []
    for ten in tensors:
        if combine_type == 'or':
            for filt in filters:
                if filt in ten.name:
                    f_tens.append(ten)
                    break
        elif combine_type == 'and':
            all_pass = True
            for filt in filters:
                if filt not in ten.name:
                    all_pass = False
                    break
            if all_pass:
                f_tens.append(ten)
    return f_tens


def global_variables(filters=None, combine_type='or'):
    global_vars = tf.global_variables()
    if filters is None:
        return global_vars
    else:
        return tensors_filter(global_vars, filters, combine_type)


def trainable_variables(filters=None, combine_type='or'):
    t_var = tf.trainable_variables()
    if filters is None:
        return t_var
    else:
        return tensors_filter(t_var, filters, combine_type)


def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """Return a Session with simple config."""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)

def print_tensor(tensors):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for i, tensor in enumerate(tensors):
        ctype = str(type(tensor))
        if 'Tensor' in ctype:
            type_name = 'Tensor'
        elif 'Variable' in ctype:
            type_name = 'Variable'
        else:
            raise Exception('Not a Tensor or Variable!')
        print(str(i) + (': %s("%s", shape=%s, dtype=%s, device=%s)'
                        % (type_name, tensor.name, str(tensor.get_shape()),
                           tensor.dtype.name, tensor.device)))
prt = print_tensor

def shape(tensor):
    sp = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in sp]

def counter(start=0, scope=None):
    with tf.variable_scope(scope, 'counter'):
        counter = tf.get_variable(name='counter',
                                  initializer=tf.constant_initializer(start),
                                  shape=(),
                                  dtype=tf.int64)
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt

def save(sess, saver, checkpoint_dir, task, model_name, counter):
    model_dir = "%s" % (task)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("Model is saving...")
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)
    print("Model has been saved...")

def load(sess, saver, checkpoint_dir, task):
    print("Reading checkpoint...")
    model_dir = "%s" % (task)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("Reading checkpoint successful...")
        return True
    else:
        return False