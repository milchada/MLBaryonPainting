#code based on: https://github.com/bnsreenu/python_for_microscopists/blob/master/090a-autoencoder_colorize_V0.2.py

#inputs need to be normalized between 0 and 1
#or -1 to 1 if tanh activation 
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, image_utils
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

blur = False

# M = np.load('dm_mass_proj.npy') #shape should be Ncluster, xsize, ysize, ndim
# V = np.load('dm_vmag_proj.npy') 
# y = np.load('gas_em_proj.npy')
M = np.load('dmmass.npy')#dm_mass_proj.npy') #shape should be Ncluster, xsize, ysize, ndim
V = np.load('dmvel.npy')#_velmag_proj.npy')
Lx = np.load('sb_gas.npy')#gas_sb_proj.npy')
kT = np.load('kt_gas.npy')#gas_kT_proj.npy')
bhar = np.load('bhar.npy')

M = np.log10(M)
V = np.log10(V)
Lx = np.log10(Lx)
Lx[Lx == -np.inf] = np.nan
weights = np.log10(np.load('dmmass.npy'))#_mass_proj.npy') )

#normalize everything to 0 - 1
M -= np.nanmin(M)
eps = 0.1
M += eps 
M /= np.nanmax(M)

#for v
V -= np.nanmin(V)
V += eps 
V /= np.nanmax(V)

#same for y
Lx -= np.nanmin(Lx)
Lx += eps 
Lx /= np.nanmax(Lx)

#X = M
X = np.ndarray((*M.shape, 2))
X[:,:,:,0] = M 
X[:,:,:,1] = V
# X = np.expand_dims(M, -1)
Lx = np.expand_dims(Lx, -1)
X[np.isnan(X)] = 0
Lx[np.isnan(Lx)] = 0

#smooth y before training. 
#but with what width? This is in kpc, whereas really it would be arcseconds. Do I use a range of values? Do I train separately for each redshift?
#what is the current resolution?? 512 pix for a cutout that's.. 8 Mpc across. So 15.625 kpc/pix.
#that's not that sharp actually. It's 2.5" at z = 0.5, or 2" at z = 1 !
#well, eROSITA is even blurrier, 15-25". So let's convolve by 5-10 of these pixels.
if blur:
	yblur = np.zeros(y.shape)
	for i in range(len(y)):
		yblur[i] = gaussian_filter(y[i], sigma=7, order=0, mode='nearest')

	y = yblur/yblur.max()
	del(yblur)
else:
	y /= np.nanmax(y)

weights[np.isnan(weights)] = 1 #the others are in the range 36-42; if this is 0, the empty regions come out all wrong. 

seed = np.random.RandomState(42) #this way it'll always be the same clusters 
inds = np.random.randint(0, len(X), size=len(X))
train = inds[:round(len(X)*.7)]
validate = inds[round(len(X)*.7):round(len(X)*.9)]
test = inds[round(len(X)*.9):]

#70/20/10 training/validate/test split
X_train = X[train]
X_valid = X[validate]
X_test  = X[test]
y_train = Lx[train]
y_valid = Lx[validate]
y_test  = Lx[test]

weights_train = weights[train].reshape((*y_train.shape))

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

def build_unet(input_shape):
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

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

model = build_unet(input_shape)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="mse",
                  metrics=["accuracy", "loss"])
# es_callback = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train,y_train,epochs=350, batch_size=16, validation_data=(X_valid, y_valid), 
	sample_weight=weights_train) #callbacks=es_callback, 

#try decreasing learning rate

model.save('autoencoder_weighted_withvmag.model')

#you know, maybe this is helpful. Maybe just rho_DM is not enough to predict gas_EM.
#what if the input has 2 channels, including vdisp_DM
#next, try VAE: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/