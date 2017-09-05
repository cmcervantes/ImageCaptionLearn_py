import math
import tensorflow as tf


def get_weights(shape, init_type='xavier', trainable=True):
    """
    Returns the tensorflow variable for a weight tensor
    :param shape: Number of dimensions (for standard problems, this is
                  typically the number of hidden nodes x number of classes)
    :param init_type:  Type of initialization (uniform, normal, xavier)
    :param trainable:  Whether this vector is trainable
    :return:           Tensorflow variable for the weight tensor
    """

    if init_type == 'xavier':
        init_range = math.sqrt(6.0 / sum(shape))
        return tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                                             seed=20170801), trainable=trainable)
    elif init_type == "normal":
        return tf.Variable(tf.random_normal(shape, stddev=1.0 / math.sqrt(shape[0]),
                                            seed=20170801), trainable=trainable)
    elif init_type == "uniform":
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05,
                                             seed=20170801), trainable=trainable)
    return None
#enddef


def get_biases(shape, init_type="xavier", trainable=True):
    """
    Returns a tensorflow variable for the model's biases
    :param shape: Number of dimensions (for standard problems, this is
                  typically the number of classes)
    :param init_type:  Type of initialization (uniform, normal, xavier)
    :param trainable: Whether this variable is trainable
    :return: Tensorflow variable for the bias tensor
    """

    if init_type == 'xavier':
        init_range = math.sqrt(6.0 / sum(shape))
        return tf.Variable(tf.random_uniform(shape, minval=-init_range,
                                             maxval=init_range, seed=20170801),
                                             trainable=trainable)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape, stddev=1.0 / math.sqrt(shape[0]),
                                            seed=20170801), trainable = trainable)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05,
                                             seed=20170801), trainable=trainable)
    return None
#enddef

