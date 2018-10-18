# -*- coding: utf-8 -*-
# @Time    : 18-10-18 下午6:07
# @Author  : Leaves
# @File    : AlexNet.py
# @Software: PyCharm

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
    #conv1
    with tf.name_scope('conv1') as scope: #将scope内生成的Variable自动命名为
        # conv1/XXX，区分不同卷积层之间的组件
        kernel = tf.Variable(tf.truncated_normal([11,11,3,96],dtype=tf.float32,stddev=1e-1),name='weight')
        #conv1的核是11*11，3是输入channel数（后面的函数run_benchmark()中指定了image的channel是3）,
        # 96是输出channel数.stddev标准差0.1
        conv = tf.nn.con2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32),trainable=Ture,name='biase')
        #初始化biases，一维向量，96个元素，全为0
        bias = tf.nn.bias_add(conv,biase) #把初始biases加到conv
        conv1 = tf