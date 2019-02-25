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


def preprocess_videos(paths, shape_r, shape_c, frames=3000):

    cap = cv2.VideoCapture(paths)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # re_num_b_s += math.ceil(nframes / b_s)

    nframes = min(nframes,frames)
    ims = np.zeros((nframes, shape_r, shape_c, 3))
    for idx_frame in range(nframes):
        ret, frame = cap.read()
        padded_frame = padding(frame, shape_r, shape_c, 3)
        ims[idx_frame] = padded_frame

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    if K.image_data_format() == 'channels_first':
        ims = ims.transpose((0, 3, 1, 2))

    return ims,nframes,height,width,fps



if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))

    width = 2048
    height = 1024

    dataset       = 'Videos'
    method_name   = 'Results_' + task_type
    st_model_path = wkdir + '/Models/model4img-' + task_type + '.h5'
    dy_model_path = wkdir + '/Models/model4vid-' + task_type + '.h5'
    output_folder = wkdir + '/DataSet/Videos/' + method_name + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    print("Build SalCNN Static Model")
    st_model = salcnn_Static_Net(img_cols=shape_c, img_rows=shape_r, img_channels=3)
    st_model.load_weights(st_model_path)

    print("Build SalCNN Smooth Model")
    dy_model = salcnn_sm_net(img_rows=shape_r_out, img_cols=shape_c_out)
    dy_model.load_weights(dy_model_path)

    vids_test_path  = wkdir + '/DataSet/Videos/Stimuli/'

    file_names = [f for f in os.listdir(vids_test_path) if (f.endswith('.avi') or f.endswith('.mp4'))]
    file_names.sort()
    nb_videos_test = len(file_names)

    for idx_video in range(nb_videos_test):

        print("%d/%d   "%(idx_video+1,nb_videos_test) + file_names[idx_video])

        ivideo_name = file_names[idx_video]
        vidframes, nframes, _, _, fps = preprocess_videos(vids_test_path + ivideo_name, shape_r, shape_c)

        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        videoWriter = cv2.VideoWriter(output_folder + ivideo_name[:-4] + '.mp4', fourcc, fps, (width, height), isColor=False)

        X_cb = preprocess_priors(nframes, shape_r_out, shape_c_out, nb_gaussian)
        X_input = [vidframes, X_cb]
        st_sal = st_model.predict(X_input, bs_st_c2d)[0]
        predictions = st_sal.copy()

        count_bs = int(nframes/nb_c3dframes)
        dy_frames = count_bs * nb_c3dframes
        dy_input = st_sal[0:dy_frames].reshape((count_bs, nb_c3dframes, shape_r_out, shape_c_out, 1))

        # dy_input = np.zeros((count_bs, nb_c3dframes, shape_r_out, shape_c_out, 1))
        # for inx_bs in range(count_bs):
        #     dy_input[inx_bs, :, :, :] = st_sal[inx_bs * nb_c3dframes:(inx_bs + 1) * nb_c3dframes]

        dy_sal = dy_model.predict(dy_input,bs_dy_c3d)
        dy_sal = dy_sal.reshape((dy_frames, shape_r_out, shape_c_out, 1))
        predictions[:dy_frames, :, :, :] = dy_sal

        # for inx_bs in range(count_bs):
        #     predictions[inx_bs * nb_c3dframes:(inx_bs + 1) * nb_c3dframes] = dy_sal[inx_bs, :, :, :]

        savepred_mat = np.zeros((nframes, height, width, 1),dtype=np.float32)
        for idx_pre, ipred in zip(range(nframes), predictions):
            isalmap = postprocess_predictions(ipred, height, width)
            savepred_mat[idx_pre,:,:,0] = isalmap

            videoWriter.write(np.uint8(isalmap))

        with open(output_folder + ivideo_name[:-4] + '_2048x1024x' + str(nframes) + '_32b.bin', "wb") as f:     #changed for the required formats
            f.write(savepred_mat.astype(np.float32))

        # savepred_mat = np.rint(savepred_mat).astype('int32')                                                  # the ".mat" files are not required, so we comment this parts
        # output_path = output_folder + ivideo_name[:-4] + '.mat'
        # h5io.savemat(output_path, {'salmap': savepred_mat})

        videoWriter.release()

print("Done ..")


