from __future__ import division

import os, cv2, sys, re
import numpy as np

import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import tensorflow as tf

from zk_config import *
from zk_utilities import *
from zk_models import *


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]
    images.sort()

    counter = 0
    while True:
        X_img = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        X_cb = preprocess_priors(b_s, shape_r_out, shape_c_out, nb_gaussian)
        yield [X_img, X_cb]
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))

    width = 2048
    height = 1024

    dataset       = 'Images'
    method_name   = 'Results_' + task_type
    model_path    = wkdir + '/Models/model4img-'+ task_type +'.h5'
    output_folder = wkdir + '/DataSet/Images/' + method_name + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Build Static SalCNN Model: " + task_type)
    model = salcnn_Static_Net(img_cols=shape_c, img_rows=shape_r, img_channels=3)
    model.load_weights(model_path)

    # for many image saliency prediction
    imgs_test_path = wkdir + '/DataSet/Images/Stimuli/'
    file_names = [f for f in os.listdir(imgs_test_path) if (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]
    file_names.sort()
    nb_imgs_test = len(file_names)

    print("Predict saliency maps for " + imgs_test_path)
    predictions = model.predict_generator(generator_test(b_s=bs_st_c2d, imgs_test_path=imgs_test_path), math.ceil(nb_imgs_test / bs_st_c2d))

    if with_CB:
        cmap = cv2.imread(wkdir + '/' + task_type + '_CB.png', -1)
    get_file_info = re.compile("(\w+\d{1,2})_(\d+)x(\d+)")
    for pred, imgname in zip(predictions[0], file_names):
        name, _, _ = get_file_info.findall(imgname.split(os.sep)[-1])[0]
        predimg = pred[:, :, 0]

        res = postprocess_predictions(predimg, height, width)
        cv2.imwrite(output_folder + name + '_woCB.png', res.astype(int))
        if with_CB:
            res = addCB(res,cmap)
        cv2.imwrite(output_folder + name + '.png', res.astype(int))

        with open(output_folder + name + '.bin', "wb") as f:
            f.write(res)


    # for single image saliency prediction
    # image_path = wkdir + '/DataSet/Images/Stimuli/P28_4000x2000.jpg'
    # X_img = preprocess_images([image_path], shape_r, shape_c)
    # X_cb = preprocess_priors(1, shape_r_out, shape_c_out, nb_gaussian)
    # X_input = [X_img, X_cb]
    # prediction = model.predict(X_input,1)[0]
    #
    # get_file_info = re.compile("(\w+\d{1,2})_(\d+)x(\d+)")
    # name, width, height = get_file_info.findall(image_path.split(os.sep)[-1])[0]
    # res = postprocess_predictions(prediction[0], int(height), int(width))
    # cv2.imwrite(output_folder + name + '.png', res.astype(int))

