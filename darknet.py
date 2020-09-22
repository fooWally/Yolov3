#############################
#                           #
#   darknet53 structure     #
#                           #
#############################
import tensorflow as tf
from tensorflow.keras import layers

def conv2d(x, filters, kernel_size, downsample=False, activate=True, bn=True):
    if downsample:
        # To reduce size of x : 256 --> 128, introduce a zero-padding
        #x = layers.ZeroPadding2D(((1,0),(1,0)))(x)
        x = layers.ZeroPadding2D(((1,1),(1,1)))(x)
        padding = 'valid'; strides = 2
    else:
        padding = 'same'; strides = 1
        
    conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,use_bias=not bn)(x)
    if bn: conv = layers.BatchNormalization()(conv)
    if activate == True : conv = layers.LeakyReLU(alpha=0.1)(conv)
    return conv

def residual_block(x, filters1, filters2):
    stored_x = x
    conv = conv2d(   x, filters1, kernel_size=1)
    conv = conv2d(conv, filters2, kernel_size=3)
    residual_x = stored_x + conv
    return residual_x


def darknet53(inputs):
    """
    conv2d (inputs, numFilters, kernel_size)
    residual_block(inputs, numFilters1, numFilters2)

    residual_1, residual_2, residual_3 will be used
    to extract features of images
    """

    x = conv2d(inputs, 32, 3) # output size: 256x256
    x = conv2d(x, 64, 3, downsample=True) # output size: 128x128

    #--------------------------------------
    #   32: filters for 1st conv layer
    #   64: filters for 2nd conv layer
    x = residual_block(x, 32, 64) # output size: 128x128

    x = conv2d(x, 128, 3, downsample=True) #output size: 64x64 downsampled from 128x128

    #--------------------------------------
    #   ( 64: filters for 1st conv layer
    #    128: filters for 2nd conv layer) x 2
    for i in range(2):
        x = residual_block(x, 64, 128)
        # final output size: 64x64
        
    x = conv2d(x, 256, 3, downsample=True) #output size: 32x32 downsampled 

    #---------------------------------------
    #   (128: filters for 1st conv layer
    #    256: filters for 2nd conv layer) x 8
    for i in range(8):
        x = residual_block(x, 128, 256)   #output size: 32x32
    residual_1 = x # 52X52

    x = conv2d(x, 512, 3, downsample=True) #output size: 16x16 downsampled

    #---------------------------------------
    #   (256: filters for 1st conv layer
    #    512: filters for 2nd conv layer) x 8
    for i in range(8):
        x = residual_block(x, 256, 512)  #output size: 16x16
    residual_2 = x # 26X26

    x = conv2d(x, 1024, 3, downsample=True) #output size: 8x8 downsampled

    #---------------------------------------
    #   (512 : filters for 1st conv layer
    #    1024: filters for 2nd conv layer) x 4
    for i in range(4):
        x = residual_block(x, 512, 1024)  #output size: 8x8
        
    #residual_3 = x # 13X13
    return residual_1, residual_2, x


##################################
#
#   TEST darknet53
#
##input_size = 416 
##input_shape = [input_size, input_size, 3]
##inputs = layers.Input(shape=input_shape)
##print('inputs =', inputs)
##
##a,b,c = darknet53(inputs)
##print(a)    # 52X52
##print(b)    # 26X26
##print(c)    # 13X13
