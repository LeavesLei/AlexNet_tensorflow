# -*- coding: utf-8 -*-
# @Time    : 18-10-18 下午6:07
# @Author  : Leaves
# @File    : AlexNet.py
# @Software: PyCharm

from datetime import datetime
import math
import time
import tensorflow as tf

from data import get_data_set

#此函数为输出当前层的参数（层名称，输出尺寸）
def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())

#此函数搭建AlexNet网络
def AlexNet(image_train_x,test=False):
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

    # fc1
    with tf.name_scope('fc1') as scope:
        # 一维化
        reshape = tf.reshape(pool5,[batch_size,-1],name='reshape')
        dim = reshape.get_shape()[1].value
        weight = tf.Variable(tf.truncated_narmal([dim,4096],dtype=tf.float32,stddev=1e-1),name='weights')
        biase = tf.Variable(tf.truncated_narmal([4096],dtype=tf.float.32,stddev=1e-1),name='biases')
        fc1 = tf.nn.relu(tf.matmul(reshape,weight)+biase,name=scope)
        if(!test):
            fc1 = tf.nn.dropout(x=fc1_full,keep_prob=0.5)
        parameter += [weight,biase]
        print_activations(fc1)
    # fc2
    with tf.name_scope('fc2') as scope:
        weight = tf.Variable(tf.truncated_narmal([4096,4096],dtype=tf.float32,stddev=1e-1),name='weights')
        biase = tf.Variable(tf.truncated_narmal([4096], dtype=tf.float.32, stddev = 1e-1), name = 'biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, weight) + biase, name=scope)
        if(!test):
            fc2 = tf.nn.dropout(x=fc2,keep_prob=0.5)
        parameter += [weight, biase]
        print_activations(fc2)
    # fc3 激活函数为softmax
    with tf.name_scope('fc3') as scope:
        weight = tf.Variable(tf.truncated_narmal([4096, 1000], dtype=tf.float32, stddev=1e-1), name='weights')
        biase = tf.Variable(tf.truncated_narmal([4096], dtype=tf.float.32, stddev = 1e-1), name = 'biases')
        #fc3 = tf.nn.softmax(tf.matmul(fc2, weight) + biase, name=scope) 后面算loss的时候会求交叉熵的
        fc3 = tf.matmul(fc2,weight) + biase
        parameter += [weight, biase]
        print_activations(fc3)
    return fc3, parameters


#求准确率函数,选用一个batch来检验，此时应该不使用dropout
def perdict(iamge_test_x,image_test_y):
    pred = AlexNet(image_test_x,test=Ture)
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(image_test_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return accuracy

# 测试函数
def run_benckmark():
    # 设置input
    image_test_x = tf.placeholder(tf.float32, [None, 3072], name='x_input')
    image_test_y = tf.placeholder(tf.float32, [None, 10], name='y_input')

    image_train_x = tf.placeholder(tf.float32, [None, 3072], name='x_input')
    image_train_y = tf.placeholder(tf.float32, [None, 10], name='y_input')
    # 构建模型
    pred, parameter = AlexNet(image_train_x)
    print('START:')
    # 设定损失函数和学习步骤
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, image_train_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # 计算成功率
    accuracy = predict(image_test_x,image_test_y)
    with tf.Session() as sess:
        init = tf.global_variable_initializer()
        #初始化变量
        sess.run(init)
        for training_round in range(num_batches):
            # 每次选取一个batch_size的样本来进行训练
            train_batch_x = train_x[training_round]
            train_batch_y = train_y[training_round]
            #开始训练
            sess.run(optimizer,feed_dict={image_train_x:train_batch_x,image_train_y:train_batch_y})
            #显示结果
            print（sess.run(accuracy,feed_dict={image_test_x:test_x,image_test_y:test_y})


# 定义batch大小与数目
batch_size = 32
num_batches = 100

learning_rate = 1e-1
# 导入数据
train_x,train_y,train_l = get_data_set(cifar=10)
test_x,test_y,test_l = get_data_set("test",cifar=10)

train_x = tf.reshape(train_x,[-1,32,32,3])
test_x = tf.reshape(test_x,[-1,32,32,3])

run_benckmark()


