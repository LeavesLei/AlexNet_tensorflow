# -*- coding: utf-8 -*-
# @Time    : 18-10-18 下午6:07
# @Author  : Leaves
# @File    : AlexNet.py
# @Software: PyCharm

import tensorflow as tf
#此函数为输出当前层的参数（层名称，输出尺寸）
def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())

#此函数搭建AlexNet网络
def inference(images):
    """
    Build the AlexNet model
    :param :
        images: Image Tensor
    :return:
        pool5: the last Tensor in the convolutional component of AlexNet #返回conv_5中池化的结果
        parameters：a list of Tensors corresponding to
        the weight and biases of the AlexNet model #返回权重和偏差列表
    """

    parameters = []
    # conv1
    with tf.name_scope('conv1') as scope: #将scope内生成的Variable自动命名为
        # conv1/XXX，区分不同卷积层之间的组件
        kernel = tf.Variable(tf.truncated_normal([11,11,3,96],dtype=tf.float32,stddev=1e-1),name='weights')
        #conv1的核是11*11，3是输入channel数（后面的函数run_benchmark()中指定了image的channel是3）,
        # 96是输出channel数.stddev标准差0.1
        conv = tf.nn.con2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32),trainable=Ture,name='biases')
        #初始化biases，一维向量，96个元素，全为0
        bias = tf.nn.bias_add(conv,biase) #把初始biases加到conv
        conv1 = tf.nn.relu(bias,name=scope)
        print_activations(conv1) #输出当前层参数
        parameter += [kernel,biases] #更新权重和偏差

    # 考虑到LRN层效果不明显，而且会让forward和backwood的速度大大下降，所以没有用LRN层
    # pool1
    pool1 = tf.nn.max_pool(conv1,
                          ksize=[1,3,3,1],
                          strides=[1,2,2,1],
                          padding='VALID',
                          name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,96,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)
    return pool5, parameters
