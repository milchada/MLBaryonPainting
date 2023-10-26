
#code based on: https://github.com/bnsreenu/python_for_microscopists/blob/master/090a-autoencoder_colorize_V0.2.py

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Multiply
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, image_utils
from keras.callbacks import EarlyStopping
import numpy as np
import gc
import tensorflow as tf

X = np.expand_dims(np.load('dm-normed-minmax.npy'), -1)
outfile = 'lx-normed-minmax.npy' #temp
y = np.expand_dims(np.load(outfile), -1)

import pandas as pd 
grp = pd.read_csv('groupcat-fp.csv')
snap = 99.
sub = grp[grp['snap'] == snap]
imin = int(sub.index.min()*3)
imax = int(sub.index.max()*3)

X = X[imin:imax]
y = y[imin:imax]

if 'minmax' in outfile:
    activation = 'sigmoid'
if 'sigma' in outfile:
    activation = 'linear'

X[np.isnan(X)] = 0 #throws out <0.2% of non-0 points; can't have NaN in image during training
y[np.isnan(y)] = 0
X[X<0] = 0
y[y<0] = 0

#80/10/10 training/validate/test split
seed = np.random.RandomState(42) #this way it'll always be the same clusters 
inds = seed.randint(0, len(X), size=len(X))
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

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

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

def build_unet(input_shape, output_channels=1, activation=activation):
#    inputs = Input(input_shape)
    input1 = Input(shape=input_shape, name='image_input')
    input2 = Input(shape=input_shape, name='masks')
    
    s1, p1 = encoder_block(input1, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    #outputs = Conv2D(output_channels, 1, padding="same", activation="linear")(d4)
    #model = Model(inputs, outputs, name="U-Net")
    
    foo = Conv2D(output_channels, 1, padding="same", activation=activation)(d4)
    outputs = Multiply()([input2, foo])
    model = Model([input1, input2], outputs, name="U-Net")

    return model

from keras.callbacks import ModelCheckpoint

filepath = "mass_in_%s_out_mse-loss_%s-norm-mask-snap%d.h5" % (outfile.split('-normed')[0], outfile.split('-')[-1].split('.')[0], snap)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model = build_unet(input_shape, nchanels)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse', #you can also use a custom-written loss function if you prefer, but MSE is common practice for regression tasks
                  metrics=["accuracy"])
mask_train = (X_train > 0).astype(int)
mask_valid = (X_valid > 0).astype(int)

history = model.fit([X_train,mask_train],y_train,epochs=100, batch_size=8, validation_data=([X_valid,mask_valid], y_valid), 
        callbacks=callbacks_list)#, sample_weight = weights)

model.save(filepath.replace('h5','model'))

import pickle
with open(filepath.replace('h5', 'history'), 'wb') as f:
    pickle.dump(history.history, f)
