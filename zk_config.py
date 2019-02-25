import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# replace the wkdir to your path
wkdir = 'E:/Salient360_2018/Task1_2'
# task_type : 'H', 'HE'
task_type = 'HE'
# whether use center bias for image saliency
with_CB = True


# batch size
bs_st_c2d = 1
bs_dy_c3d = 1
# number of rows of input images
shape_r = 640
# number of cols of input images
shape_c = shape_r*2
# number of rows of model outputs
shape_r_out = int(shape_r/8)
# number of cols of model outputs
shape_c_out = int(shape_c/8)
# number of epochs
epochs = 20
# number of learned priors
nb_gaussian = 16
# number of frames input conv3d
nb_c3dframes = 5