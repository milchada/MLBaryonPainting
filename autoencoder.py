
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
# model.summary()

def loss(yt, yp): #i want to minimise MSE but also KLD
    mse = tf.keras.losses.MeanSquaredError()
    kl = tf.keras.losses.KLDivergence()
    loss = mse(yt, yp)+abs(kl(yt, yp)) #KLD is asymmetric; try switching yp and yt
                                       #or try KLD((yt, yp) + (yp, yt))/2
    return loss

model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse', #loss_weights = weights, #loss = 'mse'
                  metrics=["accuracy"])
mask_train = (X_train > 0).astype(int)
mask_valid = (X_valid > 0).astype(int)

history = model.fit([X_train,mask_train],y_train,epochs=100, batch_size=8, validation_data=([X_valid,mask_valid], y_valid), 
        callbacks=callbacks_list)#, sample_weight = weights)

model.save(filepath.replace('h5','model'))

import pickle
with open(filepath.replace('h5', 'history'), 'wb') as f:
    pickle.dump(history.history, f)

mask_test = (X_test > 0).astype(int)
y_pred = model.predict((X_test, mask_test))

basePath = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/highres'
df = pd.read_csv(basePath.replace('highres','model-normalization.csv'))

def model_show(modelname, X_test, y_test, y_pred, tmin = 0, ntest=6, xmin = 0, xmax=512, zscale=1):
    #plot output
    os.chdir(basePath)
    norm = modelname.split('loss_')[1].split('-')[0]
    lxmin, lxmax, rhomin, rhomax, ktmin, ktmax, dmmin, dmmax = df[norm]
    # if 'kld' in modelname:
    #     X_test, y_test, y_pred = extract_model(modelname, loss='kld')
    # else:
    if 'lx' in modelname:
        label = r'$L_X$'
        cmap = cm.magma
        ymin, ymax = lxmin, lxmax
        ylab = r'erg/cm$^2$/s'
        ztix = (1e-9, 1e-7, 1e-5)
        zcut = 10
    elif 'rho' in modelname:
        label = r'$\rho_g$'
        cmap = cm.viridis
        ymin, ymax = rhomin, rhomax
        ylab = r'g/cm$^2$'
        ztix = (1e-4, 1e-3, 1e-2)
        zcut = 10
    elif 'temp' in modelname:
        label = r'$T_X$'
        cmap = cm.afmhot
        ymin, ymax = ktmin, ktmax
        ylab = 'K'
        zcut = 1
        ztix = (2, 4, 6)
    if 'nolog' in modelname:
        log=False
    else:
        log=True
    x = X_test[tmin:tmin+ntest]
    yt = y_test[tmin:tmin+ntest]
    yp = y_pred[tmin:tmin+ntest]
    err = (yp - yt)/yt
    err[np.isinf(err)] = np.nan
    
    for arr in [x, yt, yp]:
        arr[np.isnan(arr)] = 0

    x = reconstruct(x, dmmin, dmmax) * u.g.to('Msun') #projected mass 
    yt = reconstruct(yt, ymin, ymax, log=log) 
    yp = reconstruct(yp, ymin, ymax, log=log)
    
    edge = x[0,0,0]
    zmin = np.nanpercentile(yt[x>edge], zcut)
    zmax = np.nanpercentile(yt, 99.95)
    
    for arr in [yp, yt]:
        arr[x==edge] = np.nan

    if zmax/zmin > 20:
        norm = colors.LogNorm(zmin, zmax)
    else:
        norm = colors.Normalize(zmin, zmax)
    # del(X_test, y_test, y_pred)
    # gc.collect()

    
    xnorm = colors.LogNorm(np.nanmax(x)/1e4, np.nanmax(x))
    # if 'nolog' in modelname:
    #     norm = colors.Normalize(np.nanmin(yt), np.nanmax(yt))

    fig, ax = plt.subplots(nrows=4, ncols=ntest, sharex=True, sharey=True, figsize=(ntest,5))
    
    for i in range(ntest):
        im1 = ax[0][i].imshow(x[i], cmap=cm.Greys, norm = xnorm)
        im2 = ax[1][i].imshow(yt[i], cmap=cmap, norm=norm)
        im3 = ax[2][i].imshow(yp[i], cmap=cmap, norm=norm)
        im4 = ax[3][i].imshow(err[i], cmap=cm.RdBu_r, norm=colors.Normalize(-zscale,zscale))

    ax[0][0].set_ylabel('DM')
    ax[1][0].set_ylabel('%s True' % label)
    ax[2][0].set_ylabel('%s Pred' % label)
    ax[3][0].set_ylabel('(Pred - True)/True')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.tight_layout(w_pad=0, h_pad=0, rect=[0,0,.9,1])
    edges = ax[0][-1].get_position()
    l, b, w, h = edges.bounds
    
    cax1 = fig.add_axes([l+w, b, 0.1*h, h])
    edges = ax[1][-1].get_position()
    l, b, w, h = edges.bounds
    cax2 = fig.add_axes([l+w, b, 0.1*h, h])
    edges = ax[2][-1].get_position()
    l, b, w, h = edges.bounds
    cax3 = fig.add_axes([l+w, b, 0.1*h, h])
    edges = ax[3][-1].get_position()
    l, b, w, h = edges.bounds
    
    cax4 = fig.add_axes([l+w, b, 0.1*h, h])

    fig.colorbar(im1, cax=cax1, label = r'$\Sigma_{DM}/M_\odot$')
    fig.colorbar(im2, cax=cax2, label = ylab)
    fig.colorbar(im3, cax=cax3, label = ylab)
    fig.colorbar(im4, cax=cax4)
    cax2.set_yticks(ztix)
    cax3.set_yticks(ztix)
    # cax4.set_yticks([-1, -0.5, 0, 0.5, 1])
    cax2.set_ylim(zmin, zmax)
    cax3.set_ylim(zmin, zmax)
    fig.savefig(modelname.split('.')[0]+'.png', dpi=192)
    del(x, yt, yp); gc.collect()
    plt.close()

import matplotlib.pylab as plt
from matplotlib import colors, cm
import glob, os
from astropy import units as u

def reconstruct(arr, arrmin, arrmax, eps=0.1, log=True):
    # arr[arr == 0] = np.nan
    ret = arr * (arrmax-arrmin+eps)
    ret -= eps
    ret += arrmin
    if log:
        return 10**ret
    else:
        return ret

model_show(filepath, X_test, y_test, y_pred)
