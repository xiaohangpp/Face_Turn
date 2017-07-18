import tensorflow as tf
import numpy as np
import glob

class read_function(object):
	def __init__(self, batch_size, filename, label):
		self.batch_size = batch_size
		self.filename = filename
		self.label = label

	def read_png(self, filename_queue, label_queue):
		reader = tf.WholeFileReader()
		key, value = reader.read(filename_queue)
		image = tf.image.decode_png(value)
		# convert data type
		image = tf.image.convert_image_dtype(image, dtype = tf.float32)
		image = tf.image.rgb_to_grayscale(image)
		image.set_shape([182, 182, 1])

		label_reader = tf.WholeFileReader()
		key_l, value_l = label_reader.read(label_queue)
		label = tf.decode_raw(value_l, tf.int64)
		return image, label

	def input_pipeline(self, image, label):
		image_batch, image_label = tf.train.batch([image, label], batch_size = self.batch_size, capacity = self.batch_size)
		#return image_batch, tf.reshape(image_label, [self.batch_size])
		#print(image_batch.shape)
	#	print(image_label.shape)
		return image_batch, image_label



#read_data = x.read_png()
#image_batch, image_label = x.input_pipeline(read_data)

def find_all_sub(a_str, sub):
	idx = []

	start = -1
	while True:
		start = a_str.find(sub, start+1)
		if start == -1:
			break
		idx.append(start)
	return idx


def __apply__data(directory, batch_size):
	filename = list()
	label = list()
	for name in glob.glob(directory+'/*/*.png'):
		filename.append(name)
		idx = find_all_sub(name, "/")
		pose = name[idx[-2]+1:idx[-1]]
		#label.append(int(pose))
		label.append(pose)
	#print(np.array(filename).shape)
	#print(np.array(label).shape)
	x = read_function(batch_size, filename, label)
	filename_queue = tf.train.string_input_producer(filename)
	label_queue = tf.train.string_input_producer(label)
	image, label = x.read_png(filename_queue, label_queue)
	#print(np.array(image).shape)
	image_batch, image_label = x.input_pipeline(image, label)
	return image_batch, image_label
