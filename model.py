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

	with tf.variable_scope("dense1"):
		weight_layer_2 = variable([3, 3, 16, 16], name = 'w2', initialize = 0.005)

		layer_2_1 = conv(layer_1_result, weight_layer_2)

		bias_2_1 = bias_variable([16])

		layer_2_1_result = tf.nn.relu(tf.add(layer_2_1, bias_2_1))

		dense_weight_1 =  variable([3, 3, 16, 16], name = 'd1', initialize = 0.005)

		layer_2_2 = conv(layer_2_1_result, dense_weight_1)

		bias_2_2 = bias_variable([16])
		layer_2_2_result = tf.nn.relu(tf.add(layer_2_2, bias_2_2))

		layer_2_3 = conv(layer_2_2_result, weight_layer_2)
		bias_2_3 = bias_variable([16])
		layer_2_3_result = tf.nn.relu(tf.add(layer_2_3, bias_2_3))
		#_activation_summary(layer_2_1_result)
		_activation_summary(layer_2_3_result)

		layer_2_3_pool = tf.nn.max_pool(layer_2_3_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	with tf.variable_scope("conv3"):
		weight_layer_3 = variable([1, 1, 16, 64], name = 'w3', initialize = 0.005)
		layer_3 = conv(layer_2_3_pool, weight_layer_3)
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


	with tf.variable_scope("FC1"):
		flatten = tf.reshape(layer_4_pool, [-1, 128 * 46 * 46])

		full_weight1 = variable([128 * 46 * 46, 512], 'FC1', initialize = 0.005)
		perceptron1 = tf.add(tf.matmul(flatten, full_weight1), bias_variable([512]))
		layer_percep1 = tf.nn.relu(perceptron1)

		_activation_summary(layer_percep1)

	with tf.variable_scope("FC2"):
		full_weight2 = variable([512, 256], 'FC2', initialize = 0.005)
		perceptron2 = tf.add(tf.matmul(layer_percep1, full_weight2), bias_variable([256]))
		layer_percep2 = tf.nn.relu(perceptron2)
		_activation_summary(layer_percep2)

	with tf.variable_scope("FC3"):
		full_weight3 = variable([256, 128], 'FC3', initialize = 0.005)
		perceptron3 = tf.add(tf.matmul(perceptron2, full_weight3), bias_variable([128]))
		layer_percep3 = tf.nn.dropout(tf.nn.relu(perceptron3), val)

		_activation_summary(layer_percep3)

	with tf.variable_scope("FC4"):
		full_weight4 = variable([128, 13], 'FC4', initialize = 0.005)
		result = tf.add(tf.matmul(layer_percep3, full_weight4), bias_variable([13]))
		_activation_summary(result)

	softmax_result = tf.nn.softmax(result)
	#return softmax_result, result

	layer1v = layer_1_result[0:1, :, :, 0:16]
	layer2v = layer_2_pool[0:1, :, :, 0:32]
	layer3v = layer_3_result[0:1, :, :, 0:64]
	layer4v = layer_4_pool[0:1, :, :, 0:128]

	layer1v = tf.transpose(layer1v, perm=[3,1,2,0])
	layer2v = tf.transpose(layer2v, perm=[3,1,2,0])
	layer3v = tf.transpose(layer3v, perm=[3,1,2,0])
	layer4v = tf.transpose(layer4v, perm=[3,1,2,0])

	"""visualization1 = tf.concat(2, [layer1v])
	list_v1 = tf.split(0, 16, visualization1)
	visualization1 = tf.concat(1, list_v1)
	v_min = tf.reduce_min(visualization1)
	v_max = tf.reduce_max(visualization1)
	visualization1 = (visualization1-v_min)/(v_max-v_min)
	visualization1 = tf.image.convert_image_dtype(visualization1, dtype.uint8)
	tf.image_summary("filtered_images1", visualization1, max_outputs=5)"""

	visualization1 = tf.concat([layer1v, layer1v], 0)
	list_v1 = tf.split(visualization1, 16, 0)
	visualization1 = tf.concat(list_v1, 1)
	#tf.image_summary("filtered_images1", visualization1, max_outputs=512)


	visualization2 = tf.concat([layer2v, layer3v], 0)
	list_v2 = tf.split(visualization2, 16, 0)
	visualization2 = tf.concat(list_v2, 1)
    	#tf.image_summary("filtered_images2", visualization2, max_outputs=512)

	visualization3 = tf.concat([layer4v], 0)
	list_v3 = tf.split(visualization3, 16, 0)
	visualization3 = tf.concat(list_v3, 1)
	#tf.image_summary("filtered_images3", visualization3, max_outputs=512)

    	#return softmax_result, result, visualization1, visualization2, visualization3
	return softmax_result, result, visualization1, visualization2, visualization3




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
