#https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e
#import tensorflow as tf
from darknet import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras import layers, Model
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
    num_filters = (NUM_CLASS + 5)*3 
    # darknet53 as feature extractor : 52x52 grids, 26x26 grids, 13x13 grids
    x52, x26, x13 = darknet53(inputs)
    x = conv2d(x13, 512, 1)
    x = conv2d(x, 1024, 3)
    x = conv2d(x, 512, 1)
    x = conv2d(x, 1024, 3)
    x = conv2d(x, 512, 1)
    y_large = conv2d(x, 1024, 3)
    y_large = conv2d(y_large, num_filters, 1)#, activate=False, bn=False)

    x = conv2d(x, 256, 1)
    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Concatenate()([x, x26])

    x = conv2d(x, 256, 1)
    x = conv2d(x, 512, 3)
    x = conv2d(x, 256, 1)
    x = conv2d(x, 512, 3)
    x = conv2d(x, 256, 1)
    y_medium = conv2d(x, 512, 3)
    y_medium = conv2d(y_medium, num_filters, 1)#, activate=False, bn=False)

    x = conv2d(x, 128, 1)
    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Concatenate()([x, x52])

    x = conv2d(x, 128, 1)
    x = conv2d(x, 256, 3)
    x = conv2d(x, 128, 1)
    x = conv2d(x, 256, 3)
    x = conv2d(x, 128, 1)
    y_small = conv2d(x, 256, 3)
    y_small = conv2d(y_small, num_filters, 1)#, activate=False, bn=False)
    
    return [y_small, y_medium, y_large]

################################################################
#
#   TEST Yolo3(inputs)
#
input_shape = [416, 416, 3]
inputs = layers.Input(shape=input_shape)
print('inputs =', inputs)
outputs = Yolov3(inputs)
model = Model(inputs, outputs)
#model.summary()
y_small, y_medium, y_large = outputs
# num_filters = 255 = 3*(4+1+NUM_CLASSES)
print(y_small)   #(None, 52, 52, 255)
print(y_medium)  #(None, 26, 26, 255)
print(y_large)   #(None, 13, 13, 255)
