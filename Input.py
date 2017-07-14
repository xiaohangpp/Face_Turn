import tensorflow as tf
import numpy as np
import glob

class read_function(object):
	def __init__(self, batch_size, filename, label):
		self.batch_size = batch_size
		self.filename = filename
		self.label = label

	def read_png(self):
		filename_queue = tf.train.string_input_producer(self.filename)
		reader = tf.WholeFileReader()
		key, value = reader.read(filename_queue)
		image = tf.image.decode_png(value)

		return image

	def input_pipeline(self, image):
		inage_batch, image_label = tf.train.shuffle_batch([image, self.label], batch_size = self.batch_size, capacity = self.batch_size)
		return image_batch, image_label


#read_data = x.read_png()
#image_batch, image_label = x.input_pipeline(read_data)

def __apply__data(directory):
	filename = list();
	label = list();
	for name in glob.glob(directory+'/*/*.png'):
		filename.append(name)
		idx = name.find("/")
		pose = name(idx(-2:-1));
		label.append(int(pose))
	x = read_function(1, filename, label)
	image = x.read_png();
	image_batch, image_label = x.input_pipeline(image)
	return image_batch, image_label
