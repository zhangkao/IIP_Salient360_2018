from __future__ import print_function

import numpy as np
import warnings

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Activation, Conv3D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Concatenate
from keras.regularizers import l2
from keras.preprocessing import image
from keras.utils import layer_utils, conv_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers.core import Layer, InputSpec
from keras import constraints, regularizers, initializers, activations


from zk_config import *
from salcnn_vgg16 import *

from keras.legacy import interfaces
import tensorflow as tf

def resize_images_bilinear(x, height_factor, width_factor, data_format):
	"""Resizes the images contained in a 4D tensor.

	# Arguments
		x: Tensor or variable to resize.
		height_factor: Positive integer.
		width_factor: Positive integer.
		data_format: string, `"channels_last"` or `"channels_first"`.

	# Returns
		A tensor.

	# Raises
		ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
	"""
	if data_format == 'channels_first':
		original_shape = K.int_shape(x)
		new_shape = tf.shape(x)[2:]
		new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
		x = K.permute_dimensions(x, [0, 2, 3, 1])
		x = tf.image.resize_bilinear(x, new_shape)
		x = K.permute_dimensions(x, [0, 3, 1, 2])
		x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
		             original_shape[3] * width_factor if original_shape[3] is not None else None))
		return x
	elif data_format == 'channels_last':
		original_shape = K.int_shape(x)
		new_shape = tf.shape(x)[1:3]
		new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
		x = tf.image.resize_bilinear(x, new_shape)
		x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
		             original_shape[2] * width_factor if original_shape[2] is not None else None, None))
		return x
	else:
		raise ValueError('Invalid data_format:', data_format)

class UpSampling2D(Layer):

	@interfaces.legacy_upsampling2d_support
	def __init__(self, size=(2, 2), data_format=None, **kwargs):
		super(UpSampling2D, self).__init__(**kwargs)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.size = conv_utils.normalize_tuple(size, 2, 'size')
		self.input_spec = InputSpec(ndim=4)

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
			width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
			return (input_shape[0],
			        input_shape[1],
			        height,
			        width)
		elif self.data_format == 'channels_last':
			height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
			width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
			return (input_shape[0],
			        height,
			        width,
			        input_shape[3])

	def call(self, inputs):
		return resize_images_bilinear(inputs, self.size[0], self.size[1],
		                              self.data_format)

	def get_config(self):
		config = {'size': self.size,
		          'data_format': self.data_format}
		base_config = super(UpSampling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


def salcnn_SF_Net(img_rows=480, img_cols=640, img_channels=3):

	sal_input = Input(shape=(img_rows, img_cols, img_channels))
	input_shape = (img_rows, img_cols, img_channels)

	cnn = salcnn_VGG16(include_top=False, weights='imagenet', input_tensor=sal_input, input_shape=input_shape)
	# C2 = cnn.get_layer(name='block2_pool').output
	C3 = cnn.get_layer(name='block3_pool').output
	C4 = cnn.get_layer(name='block4_pool').output
	C5 = cnn.get_layer(name='block5_conv3').output

	# C2_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c2')(C2)
	C3_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c3')(C3)
	C4_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c4')(C4)
	C5_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c5')(C5)

	C5_1_up = UpSampling2D((2, 2), name='sal_fpn_p5_up')(C5_1)
	C4_1_up = UpSampling2D((2, 2), name='sal_fpn_p4_up')(C4_1)
	x = layers.concatenate([C3_1, C4_1_up, C5_1_up], axis=-1, name='sal_fpn_merge_concat')

	model = Model(inputs=[sal_input], outputs=[x], name='salcnn_sf_fpn')

	return model

def salcnn_Static_Net(img_rows=480, img_cols=640, img_channels=3):

	sfnet = salcnn_SF_Net(img_rows=img_rows, img_cols=img_cols, img_channels=img_channels)
	x = sfnet.output

	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_2')(x)

	cb_input = Input(shape=(shape_r_out, shape_c_out, nb_gaussian))
	cb_x     = Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_cb_conv2d_1')(cb_input)
	priors   = Conv2D(128, (3, 3), activation='relu', padding='same', name='sal_cb_conv2d_2')(cb_x)

	x = layers.concatenate([x, priors], axis=-1)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_cb')(x)
	x = Conv2D(1, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_3')(x)

	model = Model(inputs=[sfnet.input,cb_input], outputs=[x, x, x], name='salcnn_st_net')

	# model.summary()
	return model


def salcnn_sm_net(img_rows=60, img_cols=80):

	input_dy_c3d = Input(shape=(nb_c3dframes, img_rows, img_cols, 1))

	# type1
	x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='sal_conv3d_1')(input_dy_c3d)
	x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='sal_conv3d_2')(x)
	x = Conv3D(1, (3, 3, 3), activation='relu', padding='same', name='sal_conv3d_3')(x)

	model = Model(inputs=[input_dy_c3d], outputs=[x], name='salcnn_sm_c3d')

	model.summary()
	return model

