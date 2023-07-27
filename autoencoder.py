#code based on: https://github.com/bnsreenu/python_for_microscopists/blob/master/090a-autoencoder_colorize_V0.2.py

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, LeakyReLU, Multiply
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, image_utils
from keras.callbacks import EarlyStopping
import numpy as np
import gc
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

X = np.expand_dims(np.load('dm-normed-4sigma.npy'), -1)
outfile = 'rho-normed-4sigma.npy' #temp, lx
y = np.expand_dims(np.load(outfile), -1)

if 'minmax' in outfile:
    activation = 'sigmoid'
if 'sigma' in outfile:
    activation = 'linear'

# eps = 0.001
X[np.isnan(X)] = 0  #can't have NaN in image during training
y[np.isnan(y)] = 0
# X[X < 0] = 0 #this only drops 0.05% of valid pixels
# y[y<0] = 0   #drops 0.2% of valid pixels

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
    # inputs = Input(input_shape)
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
    # outputs = Conv2D(output_channels, 1, padding="same", activation=activation)(d4) 
    # model = Model(inputs, outputs, name="U-Net")
    foo = Conv2D(output_channels, 1, padding="same", activation=activation)(d4) 
    outputs = Multiply()([input2, foo])
    model = Model([input1, input2], outputs, name="U-Net")
    return model

filepath = "mass_in_%s_out_kld_loss.h5" % outfile.split('-')[0]
# filepath = modelnames[1].replace('model','h5')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model = build_unet(input_shape, nchanels)
# model.summary()

def tot_loss(yt, yp): #i want to minimise MSE but also KLD
    mse = tf.keras.losses.MeanSquaredError()
    kl = tf.keras.losses.KLDivergence()
    loss = mse(yt, yp)+abs(kl(yt, yp)) #KLD is asymmetric; try switching yp and yt
                                       #or try KLD((yt, yp) + (yp, yt))/2
    return loss

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tot_loss) #loss_weights = weights, #loss = 'mse')

mask_train = (X_train > 0).astype(int)
mask_valid = (X_valid > 0).astype(int)

history = model.fit([X_train, mask_train],y_train,epochs=100, batch_size=8, validation_data=([X_valid, mask_valid], y_valid), #so data is all in [0,1]
        callbacks=callbacks_list)                                                        #and it doesn't mess with activation functions

model.save(filepath.replace('h5','model'))

import pickle
with open(filepath.replace('h5', 'history'), 'wb') as f:
    pickle.dump(history.history, f)

'''
So in fact for VAEs, the loss function is defined as mse + kld
but these two have very different normalisations, so weighting them is a problem
see this thread: https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
define a new loss function accordingly