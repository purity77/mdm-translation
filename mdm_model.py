#!/usr/bin/python
# -*- coding:utf-8 -
from functools import partial

import slim
import tensorflow as tf
import data_provider
import utils

from slim import ops
from slim import scopes


def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        # sqrt开根号 reduce_sum降维压缩求和
        # reduce_mean（tensor,axis=0）降维求平均，axis轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：
        # 第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)  # tf.shape 返回tensor尺寸
    # tf.image.resize_bilinear 使用双线性插值法调整images为size images需为4维[batch,height,width,channels]
    # tf.expand_dims(input, axis=None, name=None, dim=None) 在某个维度增加个单位1的尺寸
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio


def normalized_rmse(pred, gt_truth):
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :]) ** 2), 1))  # 1水平方向求和

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)  # 第3个维度降维


def conv_model(inputs, is_training=True, scope=''):
    # summaries or losses.
    net = {}

    with tf.name_scope(scope, 'mdm_conv', [inputs]):  # 给下面op_name 加前缀mdm_conv 用with 语句解决资源释放问题
        with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
            with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
                net['conv_1'] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1')
                net['pool_1'] = ops.max_pool(net['conv_1'], [2, 2])
                net['conv_2'] = ops.conv2d(net['pool_1'], 32, [3, 3], scope='conv_2')
                net['pool_2'] = ops.max_pool(net['conv_2'], [2, 2])
                # 两个卷积层 每层32个过滤器 3*3核
                # 每层卷积层后有一个2*2 的最大池化层
                crop_size = net['pool_2'].get_shape().as_list()[1:3]
                net['conv_2_cropped'] = utils.get_central_crop(net['conv_2'], box=crop_size)
                # 中央作物的激活与第二池化层的输出，
                # 通过跳转链接concat连接起来，以保留更多相关本地信息，否则使用max池化层会丢失这些信息
                net['concat'] = tf.concat([net['conv_2_cropped'], net['pool_2']], 3)  # axis=3
    return net


def model(images, inits, num_iterations=5, num_patches=68, patch_shape=(30, 30), num_channels=3):
    #  batch_size=4 根据定义的大小
    batch_size = images.get_shape().as_list()[0]  # tensor 通过get_shape()返回张量大小，类型是元组，通过as_list 转化层list

    hidden_state = tf.zeros((batch_size, 512))  # 维度512
    dx = tf.zeros((batch_size, num_patches, 2))  # 设置一个所有元素为0的tensor,形状是括号里面的元组(4,68,2) 模型中的增量
    endpoints = {}
    dxs = []

    # zero_out_module = tf.load_op_library('zero_out.so')
    # with tf.Session(''):
    #  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()
    m_module = tf.load_op_library('./extract_patches.so')

    for step in range(num_iterations):
        with tf.device('/cpu:0'):
            patches = m_module.extract_patches(images, tf.constant(patch_shape), inits + dx)
        #  将patches 转换形状为后面参数所设置的形状(272,15,15,3) 272=4*68
        patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
        endpoints['patches'] = patches

        with tf.variable_scope('convnet', reuse=step > 0):
            net = conv_model(patches)
            ims = net['concat']

        ims = tf.reshape(ims, (batch_size, -1)) #-1 根据前面的参数,自动计算 (4,17408)

        with tf.variable_scope('rnn', reuse=step > 0) as scope:
            hidden_state = slim.ops.fc(tf.concat([ims, hidden_state], 1), 512, activation=tf.tanh)  #  tf.concat(concat_dim, values, name='concat') 连接两个矩阵
            prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
            endpoints['prediction'] = prediction
        prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
        dx += prediction
        dxs.append(dx)

    return inits + dx, dxs, endpoints
