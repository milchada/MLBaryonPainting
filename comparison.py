import matplotlib.pylab as plt
from matplotlib import colors, cm
import random
import os, gc
import numpy as np
from keras.utils import to_categorical
from keras.metrics import MeanIoU

M = np.load('dm_mass_proj.npy') #shape should be Ncluster, xsize, ysize, ndim
V = np.load('dm_velmag_proj.npy')
Lx = np.load('gas_sb_proj.npy')
kT = np.load('gas_kT_proj.npy')

def norm(arr, eps=0.1):
    arr = np.log10(arr)
    arr -= np.nanmin(arr)
    arr += eps
    arr /= np.nanmax(arr)
    arr[np.isnan(arr)] = 0
    return arr 

M = norm(M)
V = norm(V)
Lx = norm(Lx)
kT = norm(kT) 

def plot_inputs():
    #plot hist of values
    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
    for (arr, a) in zip([M, V, Lx, kT], ax.flatten()):
        a.hist(arr, range=(0,1), bins=100)
        a.set_yscale('log')
    ax[0][0].set_title('M')
    ax[0][1].set_title('V')
    ax[1][0].set_title('L$_x$')
    ax[1][1].set_title('kT')
    return fig, ax

seed = np.random.RandomState(42) #this way it'll always be the same clusters 
inds = np.random.randint(0, len(X), size=len(X))
train = inds[:round(len(X)*.7)]
validate = inds[round(len(X)*.7):round(len(X)*.9)]
test = inds[round(len(X)*.9):]

X = np.ndarray(*M.shape, 2)
y = np.ndarray(*M.shape, 2)
X[:,:,:,0] = M 
X[:,:,:,1] = V 
y[:,:,:,0] = Lx 
y[:,:,:,1] = kT 

#70/20/10 training/validate/test split
X_train = X[train]
y_train = y[train]
X_valid = X[validate]
X_test  = X[test]
y_valid = y[validate]
y_test  = y[test]

modelName = 'autoencoder_mse.model' #change
model = keras.models.load_model(modelName)

def model_show(model, fig=None, ax=None, tighten=False):
    #plot output
    y_pred = model.predict(X_test)
    if not fig:
        fig, ax = plt.subplots(nrows=4, ncols=7, sharex=True, sharey=True, figsize=(10,5))
    for i in range(5,12):
        ax[0][i-5].imshow(X_test[i], cmap=cm.Greys, norm=colors.Normalize(0, 1))
        ax[1][i-5].imshow(y_test[i], cmap=cm.magma, norm=colors.Normalize(0, 1))
        ax[2][i-5].imshow(y_pred[i], cmap=cm.magma, norm=colors.Normalize(0, 1))
        ax[3][i-5].imshow(y_pred[i]-y_test[i], cmap=cm.RdBu, norm=colors.Normalize(-1, 1))

    ax[0][0].set_ylabel('DM')
    ax[1][0].set_ylabel('Gas True')
    ax[2][0].set_ylabel('Gas Pred')
    ax[3][0].set_ylabel('Pred - True')
    if tighten:
        for i in range(3):
            plt.tight_layout(h_pad = 0.5)
    return fig, ax 

def model_progress(model):
    fig, ax = plt.subplots()
    plt.plot(model.history['accuracy'], label='Accuracy', color='tab:orange')
    plt.plot(model.history['val_accuracy'], label='Val accuracy', color='tab:green')
    ax2 = ax.twinx()
    ax2.plot(model.history['loss'], label='Loss', color='tab:orange', linestyle='dotted')
    ax2.plot(model.history['val_loss'], label='Val loss', color='tab:green',linestyle='dotted')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    plt.legend()
    return fig, ax 

def failures(model):
    y_pred = model.predict(X_test)
    diff_per_pix = [np.sum(yp - yt)/float(np.size(yt[yt>0])) for (yp, yt) in zip(y_pred, y_test)]
    return diff_per_pix

def powerspec(image, ax, label=False):
    import scipy.stats as stats
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    npix = image.shape[0]
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    ax.loglog(kvals, Abins)
    if label:
        ax.set_xlabel("$k$")
        ax.set_ylabel("$P(k)$")

def agn_lbol(snap, halo):
    GroupFirstSub = il.groupcat.loadHalos(basePath,135,fields=['GroupBHMdot'])