#https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e
import tensorflow as tf
from darknet import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers
clear_session()
#########################################
#                                       #
#   Yolov3 as multi-scale detector      #
#                                       #
#########################################
def Yolov3(inputs):
    
    NUM_CLASS = 80 # for ms-coco dataset
    # tx,ty,tw,th + objectness + num_class for one anchor box
    # predict 3 anchor boxes for each scale. Hence multiply by 3
    num_filters = (4 + 1 + NUM_CLASS)*3 
    # darknet53 as feature extractor : 52x52 grids, 26x26 grids, 13x13 grids
    feat52, feat26, feat13 = darknet53(inputs)

    for i in range(3):
        x13 = conv2d(feat13, 512, 1)
        if i == 2: cache_x13 = x13
        x13 = conv2d(x13, 1024, 3)
        feat13 = x13
    large_feat = conv2d(x13, num_filters, 1, activate=False, bn=False)

    x13 = conv2d(x13, 256, 1)
    x13 = layers.UpSampling2D(size=(2,2))(cache_x13)
    x_concat1 = layers.Concatenate(axis=-1)([x13, feat26])

    for i in range(3):
        x26 = conv2d(x_concat1, 256, 1)
        if i == 2: cache_x26 = x26
        x26 = conv2d(x26, 512, 3)
        x_concat1 = x26
    medium_feat = conv2d(x26, num_filters, 1, activate=False, bn=False)

    x26 = layers.UpSampling2D(size=(2,2))(cache_x26)
    
    x_concat2 = layers.Concatenate(axis=-1)([x26, feat52]) # or #x_concat2 = tf.concat([x26, feat52], axis=-1)

    for i in range(3):
        x52 = conv2d(x_concat2, 128, 1)
        x52 = conv2d(x52, 256, 3)
        x_concat2 = x52
        
    small_feat = conv2d(x52, num_filters, 1, activate=False, bn=False)

    return [small_feat, medium_feat, large_feat]

############################
#
#   TEST Yolov3(inputs)
#
##input_size = 416 
##input_shape = [input_size, input_size, 3]
##inputs = layers.Input(shape=input_shape)
##print('inputs =', inputs)
##small_feat, medium_feat, large_feat = Yolov3(inputs)
##
### num_filters = 255 = 3*(4+1+NUM_CLASSES)
##print(small_feat)   #(None, 52, 52, 255)
##print(medium_feat)  #(None, 26, 26, 255)
##print(large_feat)   #(None, 13, 13, 255)
