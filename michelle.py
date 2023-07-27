#code based on: https://github.com/bnsreenu/python_for_microscopists/blob/master/090a-autoencoder_colorize_V0.2.py

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, LeakyReLU
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, image_utils
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
from keras.callbacks import EarlyStopping
import numpy as np
import gc
import tensorflow as tf
from scipy.ndimage import gaussian_filter

X = np.expand_dims(np.load('dm-normed-minmax.npy'), -1)
outfile = 'rho-normed-minmax.npy' #temp, lx
y = np.expand_dims(np.load(outfile), -1)

if 'minmax' in outfile:
    activation = 'sigmoid'
if 'sigma' in outfile:
    activation = 'linear'

# eps = 0.001
X[np.isnan(X)] = 0  #can't have NaN in image during training
y[np.isnan(y)] = 0
X[X < 0] = 0 #this only drops 0.05% of valid pixels
y[y<0] = 0   #drops 0.2% of valid pixels

#80/10/10 training/validate/test split
seed = np.random.RandomState(42) #this way it'll always be the same clusters 
inds = np.random.randint(0, len(X), size=len(X))
train = inds[:round(len(X)*.8)]
validate = inds[round(len(X)*.8):round(len(X)*.9)]
test = inds[round(len(X)*.9):]
nchanels = 1

X_train = X[train]
X_valid = X[validate]
y_train = y[train]
y_valid = y[validate]
X_test  = X[test]
y_test  = y[test]

input_shape = X_train[0].shape

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = LeakyReLU()(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = LeakyReLU()(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, output_channels=1, activation = activation):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(output_channels, 1, padding="same", activation=activation)(d4) 

    #is there a way that I can force output = 0 in the pixels where input = 0?

    model = Model(inputs, outputs, name="U-Net")
    return model
