# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>
"""
Utilities for constructing the residual neural network
"""

import pickle
import numpy as np
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.layers as layers


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        with tf.name_scope('mean'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    """Initialize weights"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initialize biases"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_timestamp():
    """Get time at present"""
    import time
    timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    return timestamp


def get_batch_norm(inputs, is_training, scope=None):
    """Do batch normalization"""
    bn_out = layers.batch_norm(
        inputs=inputs,
        center=True,
        scale=True,
        is_training=is_training,
        scope=scope
        )
    return bn_out

def get_block_en(resnet_classic_param, is_training=None):
    """Generate encoder block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3', 'depth1', 'stride'])

    for i, bottle in enumerate(resnet_classic_param):
        bottle = bottle_conf._make(bottle)
        bottle_params = []
        # layer 1
        bottle_params.append(
            ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
        # layer 2
        bottle_params.append(
            ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
        # layer 3
        if i == len(resnet_classic_param):
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
        else:
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))

        block_params.append(bottle_params)

    return block_params


def get_block_list(blocks_params, input_depth):
    """Transfer the blocks parameters into list"""
    block_list = [input_depth]
    block_stride = []
    for block in blocks_params:
        for bottle in block:
            bottle_in = bottle[1]
            bottle_out = bottle[0]
            bottle_stride = bottle[2]
            block_list.append(bottle_in)
            block_list.append(bottle_in)
            block_list.append(bottle_out)
            block_stride.append(bottle_stride)

    return block_list, block_stride


def get_block_de_params(block_list, block_stride):
    """Generate decoder block paramters from reversed encoder block list"""
    blocks_de = []
    # reverse
    block_list = block_list[0:-1]
    block_list.reverse()
    num_bottles = len(block_list) // 3
    block_stride.reverse()
    # Whether 3x
    if num_bottles != len(block_stride):
        print("Number of bottles should be equal to number of strides, please check...")
        return None
    else:
        block = []
        for i in range(num_bottles):
            bottle = [(block_list[i*3+2], block_list[i*3+0], block_stride[i])]
            block += bottle
            if i + 1 == num_bottles or block_stride[i+1] > 1: # or 的判断有先后顺序
                blocks_de.append(block)
                block = []

    return blocks_de


def get_block_de(resnet_classic_param, is_training=None):
    """Generate decoder block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3','depth1','stride'])

    for i, bottle in enumerate(resnet_classic_param):
        bottle = bottle_conf._make(bottle)
        bottle_params = []
        # layer 1
        bottle_params.append(
            ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
        # layer 2
        bottle_params.append(
            ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
        # layer 3
        if i == len(resnet_classic_param):
            # Batch_norm
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
        else:
            bottle_params.append(
                ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))
        block_params.append(bottle_params)

    return block_params

def get_block(resnet_classic_param, encode_flag=True, is_training=None):
    """Generate block parameters according to the
       classic list like structure.

    input
    =====
    resnet_classic_param: list
        a list composed of bottleneck configuration
    encode_flag: bool
        It true, encoder blosk; if false, decoder block
    is_training: tf.placeholder
        a placeholder for batch_normalization

    output
    ======
    block_params: list
        block parameters generated for class Block
    """
    block_params = []
    bottle_conf = namedtuple(
        'bottle_conf',
        ['depth3','depth1','stride'])

    if encode_flag:
        for i, bottle in enumerate(resnet_classic_param):
            bottle = bottle_conf._make(bottle)
            bottle_params = []
            # layer 1
            bottle_params.append(
                ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
            # layer 2
            bottle_params.append(
                ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
            # layer 3
            if i == len(resnet_classic_param):
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
            else:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))

            block_params.append(bottle_params)
    else:
        for i, bottle in enumerate(resnet_classic_param):
            bottle = bottle_conf._make(bottle)
            bottle_params = []
            # layer 1
            bottle_params.append(
                ((1, 1, bottle.depth1), 1, False,'SAME', tf.nn.relu))
            # layer 2
            bottle_params.append(
                ((3, 3, bottle.depth1), bottle.stride, False, 'SAME', tf.nn.relu))
            # layer 3
            if i == len(resnet_classic_param):
                # Batch_norm
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, True, 'SAME', tf.nn.relu))
            else:
                bottle_params.append(
                    ((1, 1, bottle.depth3), 1, False, 'SAME', tf.nn.relu))
            block_params.append(bottle_params)

    return block_params


def get_odd_flags(odd_flags, blocks, flag_input=False):
    """Reverse the odd_flag list from the encoder"""
    odd_list = [flag_input]
    for odd_block in odd_flags:
        odd_list += odd_block
    # reverse
    odd_list.reverse()
    odd_de = blocks.copy()
    i = 0
    for m, odd_block in enumerate(odd_de):
        odd_de[m] = blocks[m].copy()
        for n, bottle in enumerate(odd_block):
            odd_de[m][n] = odd_list[i+1]
            i += 1

    return odd_de


def gen_validation(data, valrate = 0.2, label=None):
    """Separate the dataset into training and validation subsets.

    inputs
    ======
    data: np.ndarray
        The input data, 4D matrix
    label: np.ndarray or list, opt
        The labels w.r.t. input data, optional

    outputs
    =======
    data_train: {"data": , "label": }
    data_val: {"data":, "label":}
    """
    if label is None:
        label_train = None
        label_val = None
        idx = np.random.permutation(len(data))
        num_val = int(len(data)*valrate)
        data_val = {"data": data[idx[0:num_val],:,:,:],
                    "label": label_val}
        # train
        data_train = {"data": data[idx[num_val:],:,:,:],
                      "label": label_train}
    else:
        # Training and validation
        idx = np.random.permutation(len(data))
        num_val = int(len(data)* valrate)
        data_val = {"data": data[idx[0:num_val],:,:,:],
                    "label": label[idx[0:num_val],:]}
        # train
        data_train = {"data": data[idx[num_val:],:,:,:],
                      "label": label[idx[num_val:],:]}

    return data_train,data_val


def gen_BatchIterator(data, batch_size=100, shuffle=True):
    """
    Return the next 'batch_size' examples
    from the X_in dataset

    Reference
    =========
    [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
    Input
    =====
    data: 4d np.ndarray
        The samples to be batched
    batch_size: int
        Size of a single batch.
    shuffle: bool
        Whether shuffle the indices.

    Output
    ======
    Yield a batch generator
    """
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt]


def gen_BatchIterator_label(data, label, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset
        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        label: np.ndarray
            The labels to be batched w.r.t. data
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.
        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        else:
            indices = np.arange(len(data))
        return indices



def down_dimension(code, method='tSNE', params=None):
    """
    Do dimension decreasing of the codes, so as to evaluate samples'
    distributions.
    Inputs
    ======
    code: np.ndarray
        The estimated codes by the cae net on the samples.
    method: str
        The method of dimension decreasing, could be PCA, tSNE or Kmeans,
        default as PCA.
    params: dict
        Corresponding parameters to the method.
    Output
    ======
    code_dim: np.ndarray
    The dimension decreased matrix.
    """
    if method == 'PCA':
        from sklearn.decomposition import PCA
        code_dim = PCA().fit_transform(code)
    elif method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE()
        for key in params.keys():
            try:
                setattr(tsne, key, params['key'])
            except:
                continue
        code_dim = tsne.fit_transform(code)
    elif method == 'Kmeans':
        from sklearn.cluster import KMeans
        code_dim = KMeans()
        for key in params.keys():
            try:
                setattr(code_dim, key, params['key'])
            except:
                continue
        code_dim.fit(code)
    else:
        print("The method %s is not supported at present." % method)

    return code_dim


def one_hot(y, numclass):
    y_hot = np.zeros([len(y), numclass])
    for i, s in enumerate(y):
        y_hot[i, int(s)] = s
    return y_hot


def save_net(sess, namepath, netpath, savedict):
    """Save the net"""
    import pickle
    import sys
    sys.setrecursionlimit(1000000)

    with open(namepath, 'wb') as fp:
        pickle.dump(savedict, fp)

    # save the net
    saver = tf.train.Saver()
    saver.save(sess, netpath)


def load_net(namepath):
    """
    Load the cae network

    reference
    =========
    [1] https://www.cnblogs.com/azheng333/archive/2017/06/09/6972619.html

    input
    =====
    namepath: str
        Path to save the trained network

    output
    ======
    sess: tf.Session()
        The restored session
    names: dict
        The dict saved variables names
    """
    try:
        fp = open(namepath,'rb')
    except:
        return None

    names = pickle.load(fp)

    # load the net
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, names['netpath'])

    return sess, names


def gen_norm(img):
    """Image normalization"""
    return (img-img.min())/(img.max() - img.min())


def getGMM(data, n_components=2, covariance_type='full'):
    """mixture Gaussian Model estimation

    inputs
    ======
    data: np.ndarray
        An array of the features, a sample for a row
    n_components: int
        number of GMM components, default as two
    covariance_type: str
        Type of the covariance

    return
    ======
    gmm: fitted gmm
    """
    from sklearn import mixture
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(data)
    return gmm

