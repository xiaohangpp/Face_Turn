import tensorflow as tf
import numpy as np
import time
import Input
import os, re
TOWER_NAME = 'tower'

def _activation_summary(x):
    with tf.device('/cpu:0'):
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def variable(shape, name, initialize):
	with tf.device('/gpu:0'):
		initial = tf.random_normal(shape, stddev = initialize, name = name)
		return tf.Variable(initial)

def bias_variable(shape):
	with tf.device('/cpu:0'):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

def conv(matrix, weights, padding = 'SAME'):
	with tf.device('/gpu:0'):
		return tf.nn.conv2d(matrix, weights, strides = [1, 1, 1, 1], padding = 'SAME')

def model_forward(image, val):
	with tf.variable_scope("conv1"):
		image_matrix = tf.reshape(image, [-1, 182, 182, 1])

		weight_layer_1 = variable([3, 3, 1, 16], name = 'w1', initialize = 0.005)
		layer_1 = conv(image_matrix, weight_layer_1)
		bias_1 = bias_variable([16])
		layer_1_result = tf.nn.relu(tf.add(layer_1, bias_1))

		_activation_summary(layer_1_result)

	with tf.variable_scope("conv2"):
		weight_layer_2 = variable([3, 3, 16, 32], name = 'w2', initialize = 0.005)
		layer_2 = conv(layer_1_result, weight_layer_2)
		bias_2 = bias_variable([32])
		layer_2_result = tf.nn.relu(tf.add(layer_2, bias_2))

		_activation_summary(layer_2_result)

		layer_2_pool = tf.nn.max_pool(layer_2_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	with tf.variable_scope("conv3"):
		weight_layer_3 = variable([3, 3, 32, 64], name = 'w3', initialize = 0.005)
		layer_3 = conv(layer_2_pool, weight_layer_3)
		bias_3 = bias_variable([64])
		layer_3_result = tf.nn.relu(tf.add(layer_3, bias_3))

		_activation_summary(layer_3_result)

	with tf.variable_scope("conv4"):
		weight_layer_4 = variable([3, 3, 64, 128], name = 'w4', initialize = 0.005)
		layer_4 = conv(layer_3_result, weight_layer_4)
		bias_4 = bias_variable([128])
		layer_4_result = tf.nn.relu(tf.add(layer_4, bias_4))

		_activation_summary(layer_4_result)

		layer_4_pool = tf.nn.max_pool(layer_4_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	with tf.variable_scope("conv5"):
		weight_layer_5 = variable([3, 3, 128, 128], name = 'w5', initialize = 0.005)
		layer_5 = conv(layer_4_pool, weight_layer_5)
		bias_5 = bias_variable([128])
		layer_5_result = tf.nn.relu(tf.add(layer_5, bias_5))

		_activation_summary(layer_5_result)

	with tf.variable_scope("conv6"):
		weight_layer_6 = variable([3, 3, 128, 256], name = 'w6', initialize = 0.005)
		layer_6 = conv(layer_5_result, weight_layer_6)
		bias_6 = bias_variable([256])
		layer_6_result = tf.nn.relu(tf.add(layer_6, bias_6))

		_activation_summary(layer_6_result)

	with tf.variable_scope("conv7"):
		weight_layer_7 = variable([3, 3, 256, 256], name = 'w7', initialize = 0.005)
		layer_7 = conv(layer_6_result, weight_layer_7)
		bias_7 = bias_variable([256])
		layer_7_result = tf.nn.relu(tf.add(layer_7, bias_7))

		_activation_summary(layer_7_result)

	with tf.variable_scope("conv8"):
		weight_layer_8 = variable([3, 3, 256, 512], name = 'w8', initialize = 0.005)
		layer_8 = conv(layer_7_result, weight_layer_8)
		bias_8 = bias_variable([512])
		layer_8_result = tf.nn.relu(tf.add(layer_8, bias_8))

		_activation_summary(layer_8_result)

	with tf.variable_scope("FC1"):
		flatten = tf.reshape(layer_8_result, [-1, 512 * 46 * 46])

		full_weight1 = variable([512 * 46 * 46, 512], 'FC1', initialize = 0.005)
		perceptron1 = tf.add(tf.matmul(flatten, full_weight1), bias_variable([512]))
		layer_percep1 = tf.nn.relu(perceptron1)

		_activation_summary(layer_percep1)

	with tf.variable_scope("FC2"):
		full_weight2 = variable([512, 256], 'FC2', initialize = 0.005)
		perceptron2 = tf.add(tf.matmul(layer_percep1, full_weight2), bias_variable([256]))
		layer_percep2 = tf.nn.dropout(tf.nn.relu(perceptron2), val)

		_activation_summary(layer_percep2)

	with tf.variable_scope("FC3"):
		full_weight3 = variable([256, 128], 'FC3', initialize = 0.005)
		perceptron3 = tf.add(tf.matmul(layer_percep2, full_weight3), bias_variable([128]))
		layer_percep3 = tf.nn.dropout(tf.nn.relu(perceptron3), val)

		_activation_summary(layer_percep3)

	with tf.variable_scope("FC4"):
		full_weight4 = variable([128, 13], 'FC4', initialize = 0.005)
		result = tf.add(tf.matmul(layer_percep3, full_weight4), bias_variable([13]))
		_activation_summary(result)

	softmax_result = tf.nn.softmax(result)
	return softmax_result, result

	layer1v = layer_1_result[0:1, :, :, 0:16]
	layer2v = layer_2_pool[0:1, :, :, 0:32]
	layer3v = layer_3_result[0:1, :, :, 0:64]
	layer4v = layer_4_pool[0:1, :, :, 0:128]
	layer5v = layer_5_result[0:1, :, :, 0:128]
	layer6v = layer_6_result[0:1, :, :, 0:256]
	layer7v = layer_7_result[0:1, :, :, 0:256]
	layer8v = layer_8_result[0:1, :, :, 0:512]


	layer1v = tf.transpose(layer1v, perm=[3,1,2,0])
	layer2v = tf.transpose(layer2v, perm=[3,1,2,0])
	layer3v = tf.transpose(layer3v, perm=[3,1,2,0])
	layer4v = tf.transpose(layer4v, perm=[3,1,2,0])
	layer5v = tf.transpose(layer5v, perm=[3,1,2,0])
	layer6v = tf.transpose(layer6v, perm=[3,1,2,0])
	layer7v = tf.transpose(layer7v, perm=[3,1,2,0])
	layer8v = tf.transpose(layer8v, perm=[3,1,2,0])

	"""visualization1 = tf.concat(2, [layer1v])
	list_v1 = tf.split(0, 16, visualization1)
	visualization1 = tf.concat(1, list_v1)
	tf.image_summary("filtered_images1", visualization1)

	visualization2 = tf.concat(2, [layer2v, layer3v])
	list_v2 = tf.split(0, 16, visualization2)
	visualization2 = tf.concat(1, list_v2)
	tf.image_summary("filtered_images2", visualization2)

	visualization3 = tf.concat(2, [layer4v, layer5v, layer6v, layer7v, layer8v])
	list_v3 = tf.split(0, 16, visualization3)
	visualization3 = tf.concat(1, list_v3)
	tf.image_summary("filtered_images3", visualization3)"""

def model_loss(result, labels):
	labels = tf.cast(labels, tf.int64)
	#tf.one_hot(labels, 13)
	cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = result)
	cost = tf.reduce_mean(cross_entropy_loss)
	_activation_summary(cost)
	return cost

def train(cost):
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.005
	rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
	with tf.device('/gpu:0'):
		gradient_apply = tf.train.GradientDescentOptimizer(learning_rate = rate).minimize(cost)
	return gradient_apply
