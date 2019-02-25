from __future__ import division
import cv2
import numpy as np
import keras.backend as K
import scipy.io
import scipy.ndimage
import hdf5storage as h5io
EPS = 2.2204e-16

def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i,:,:,0] = padded_map.astype(np.float32)
        ims[i,:,:,0] /= 255.0

    return ims


def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        fix_map = h5io.loadmat(path)["fixmap"]
        ims[i,:,:,0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255


def preprocess_priors(b_s, shape_r, shape_c, channels = 16):

    ims = get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims,b_s,axis=0)

    return ims

def preprocess_priors_3d(b_s, time_dims,shape_r, shape_c, channels = 16):

    ims = get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, time_dims, axis=0)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, b_s, axis=0)
    return ims

def get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,16,0)
    mu_y = np.repeat(0.5,16,0)

    # sigma_x = np.array([4 / width, 4 / width, 4 / width, 4 / width,
    #            8 / width, 8 / width, 8 / width, 8 / width,
    #            16 / width, 16 / width, 16 / width, 16 / width,
    #            32 / width, 32 / width, 32 / width, 32 / width])
    sigma_y = np.array([4 / height, 8 / height, 16 / height, 32 / height,
               4 / height, 8 / height, 16 / height, 32 / height,
               4 / height, 8 / height, 16 / height, 32 / height,
               4 / height, 8 / height, 16 / height, 32 / height])
    # sigma_x = sigma_y.transpose()
    sigma_x = np.ones((16)) * width

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + K.epsilon()) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + K.epsilon())))

    return gaussian

def normalize(x, method='standard', axis=None):
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res


def addCB(s_map,c_map):
	smap = np.array(s_map/255, copy=False)
	cmap = np.array(c_map/255, copy=False)
	salmap = 255*normalize(np.multiply(smap,cmap)+smap+cmap,'range')
	return salmap
