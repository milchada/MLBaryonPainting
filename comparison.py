import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib import colors, cm
import os, gc, glob
import numpy as np
from keras import models
import pandas as pd 
import pickle
import tensorflow as tf 
from astropy import units as u 

def loss(yt, yp): #i want to minimise MSE but also KLD
    mse = tf.keras.losses.MeanSquaredError()
    kl = tf.keras.losses.KLDivergence()
    loss = mse(yt, yp)+abs(kl(yt, yp)) #KLD is asymmetric; try switching yp and yt
                                       #or try KLD((yt, yp) + (yp, yt))/2
    return loss

modelnames = glob.glob('mass*model'); modelnames.sort()

histories = [m.replace('model','history') for m in modelnames]

basePath = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/highres'
df = pd.read_csv(basePath.replace('highres','model-normalization.csv'))

def renorm(arr, arrmin, arrmax, eps=0.1, log=True):
    if log:
        ret = np.log10(arr) #log10(Lx)
        ret[ret == -np.inf] = np.nan 
    else:
        ret = arr
        arrmin = 10**arrmin 
        arrmax = 10**arrmax
    ret -= arrmin #min(log10(Lx))
    ret += eps
    ret /= (arrmax-arrmin+eps) #max(log10(Lx))
    return ret #to reconstruct arr, need np.nanmin(arr), np.nanmax(arr)

def reconstruct(arr, arrmin, arrmax, eps=0.1, log=True):
    # arr[arr == 0] = np.nan
    ret = arr * (arrmax-arrmin+eps)
    ret -= eps
    ret += arrmin
    if log:
        return 10**ret
    else:
        return ret

os.chdir(basePath)
 #these arrays are normalised so mean - 4sigma = 0, mean + 4sigma = 1

def extract_model(modelname, fit=True, loss='mse', dmo=False, maskrad=None, mask_inside=False):
    model = models.load_model(modelname, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss)
    if fit:
        normd = modelname.split('loss_')[1].split('-')[0]
        quantity = modelname.split('in_')[1].split('_out')[0]
        if quantity == 'all':
            x = np.load('rho-normed-%s.npy' % normd)
            y = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3))
            y[:,:,:,0] = np.load('rho-normed-%s.npy' % normd)
            y[:,:,:,1] = np.load('temp-normed-%s.npy' % normd)
            y[:,:,:,2] = np.load('lx-normed-%s.npy' % normd)
            del(x)
        else:
            y = np.load('%s-normed-%s.npy' % (quantity, normd))
        if dmo:
            X_test = np.load('../%s/dm-normed-%s.npy' % (dmo, normd))
            y_test = y
        else:
            X = np.load('dm-normed-%s.npy' % normd)
            seed = np.random.RandomState(42) #this way it'll always be the same clusters 
            inds = seed.randint(0, len(X), size=len(X))
            test = inds[round(len(X)*.9):]
            X_test = X[test]
            y_test = y[test]
            del(X, y); gc.collect()
        X_test[np.isnan(X_test)] = 0
        if maskrad:
            X, Y = np.meshgrid(np.arange(512), np.arange(512))
            X -= 256
            Y -= 256
            d = np.sqrt(X**2 + Y**2)
            mask = (d > maskrad)
            if mask_inside:
                mask = ~mask
            for i in range(len(X_test)):
                X_test[i][mask] = 0
        
        if 'mask' in modelname:
            mask_test = (X_test > 0).astype(int)
            y_pred = model.predict([X_test, mask_test])
        else:
            y_pred = model.predict(X_test)
        y_test[X_test == 0] = 0 #helpful for Lx
        y_pred[X_test == 0] = 0 #this should not be necessary now but i'm scared to change it
        return X_test, y_test, y_pred
    else:
        return model

def model_show(modelname, tmin = 0, ntest=6, xmin = 0, xmax=512, zscale=1):
    #plot output
    os.chdir(basePath)
    norm = modelname.split('loss_')[1].split('-')[0]
    lxmin, lxmax, rhomin, rhomax, ktmin, ktmax, dmmin, dmmax = df[norm]
    # if 'kld' in modelname:
    #     X_test, y_test, y_pred = extract_model(modelname, loss='kld')
    # else:
    X_test, y_test, y_pred = extract_model(modelname, loss='mse')
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
    yp = y_pred[tmin:tmin+ntest,:,:,0]
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

def compare_pixels(modelnames, ncols=3, yscale='linear'):
    nrows = int(len(modelnames)/ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(9,12))
    i = 0
    for modelname in modelnames:
        if 'lx' in modelname:
            prop = r'$L_X$'
        if 'rho' in modelname:
            prop = r'$\rho_g$'
        if 'temp_nolog' in modelname:
            prop = r'$T_{g, \rm linear}$'
        if 'temp_out' in modelname:
            prop = r'$T_g$'
        err = 'MSE'
        if 'minmax' in modelname:
            nrm = 'min-max'
        else:
            nrm = r'$4-\sigma$'
        if 'mask' in modelname:
            label = prop+','+nrm+'+mask'
        else:
            label = prop+','+nrm
        X_test, y_test, y_pred = extract_model(modelname)
        y_test[y_test == 0] = np.nan
        y_pred[y_pred == 0] = np.nan
        ax.flatten()[i].hist(y_test.flatten(), range=(np.nanmin(y_test),np.nanmax(y_test)), bins=1000, label='True', histtype='step', linewidth=2)
        ax.flatten()[i].hist(y_pred.flatten(), range=(np.nanmin(y_test),np.nanmax(y_test)), bins=1000, label='Pred', histtype='step', linestyle = 'dotted', linewidth=2)
        ax.flatten()[i].set_xlabel(label, fontsize=14)
        i += 1
    plt.legend()
    plt.yscale(yscale)
    for a in ax[-1]:
        xla = a.get_xticklabels()
        a.set_xticklabels(xla, fontsize=14)
    plt.xlim(0,1)
    for a in ax[:,0]:
        a.set_yticks([5e4, 1e5, 1.5e5, 2e5])
        a.set_yticklabels([5, 10, 15, 20], fontsize=14)
    plt.tight_layout()
    return fig, ax

def models_compare_training(histories, train=True, val=True, fig=None):
    def model_progress(model, color, fig, ax, label, train=True, val=True):    
        with open(model, "rb") as file_pi:
            history = pickle.load(file_pi)
        if train:
            ax.plot(history['loss'], color=color, label=label)
        if val:
            if train:
                ls = 'dotted'
            else:
                ls = '-'
            ax.plot(history['val_loss'], color=color,linestyle=ls)
        ax.set_yscale('log')
        ax.legend()
    
    fig, ax = plt.subplots(ncols = 4, sharex=True, sharey=True, figsize=(12,4))
    for model in histories:
        prop = model.split('in_')[1].split('_out')[0]
        loss = model.split('out_')[1].split('-loss')[0]
        normd = model.split('loss_')[1].split('-')[0]
        if prop == 'rho':
            col = 0
        elif prop == 'temp':
            col = 1
        elif prop == 'temp_nolog':
            col = 2
        else:
            col = 3
        if normd == 'minmax':
            model_progress(model, 'tab:blue', fig, ax[col], train=train, val=val, label=loss)
        else:
            model_progress(model, 'tab:orange', fig, ax[col], train=train, val=val, label=normd)
    ax[0].set_title(r'$\rho_g$')
    ax[1].set_title(r'$T_g$')
    ax[2].set_title(r'$T_g$, linear training')
    ax[3].set_title(r'$L_X$')
    ax[0].set_ylabel('Loss')
    for a in ax:
        a.set_xlabel('Epoch')
    return fig, ax


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
    #maybe i could take the power spectrum of each test image, then show mean and std deviation
    #and same for the predicted images



from astropy import units as u
dx = (4000./512) * u.kpc.to('cm') #kpc/pixel --> cm/pix; lx is in photons/cm**2/s
dA = dx**2

def scaling_relations(ret=False, plot=True, type='lx'):
    if type == 'lx':
        modelname = 'mass_in_lx_out_mse-loss_minmax-norm-mask.h5'
        X_dm = np.load('../dmo/dm-normed-minmax.npy') #+ np.log10(1-fgas)
    else:
        modelname = 'mass_in_rho_out_mse-loss_minmax-norm-mask.h5'
        X_dm = np.load('../dmo/dm-normed-minmax.npy') #+ np.log10(1-fgas)
    X_dm[np.isnan(X_dm)] = 0
    seed = np.random.RandomState(42) #this way it'll always be the same clusters 
    inds = seed.randint(0, len(X_dm), size=len(X_dm))
    test = inds[round(len(X_dm)*.9):]
    X_test_dm = X_dm[test]
    del(X_dm); gc.collect()
    model = models.load_model(modelname, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    X_mask = (X_test_dm > 0).astype(int)
    y_pred_dm = model.predict([X_test_dm, X_mask])

    X_test, y_true, y_pred = extract_model(modelname)
    
    if type == 'lx':
        lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df['minmax']
        ymin, ymax = lxmin, lxmax
    else:
        lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df['minmax']
        ymin, ymax = rhomin, rhomax
    X_test = reconstruct(X_test, dmmin, dmmax)
    y_true = reconstruct(y_true, ymin, ymax)
    y_pred = reconstruct(y_pred, ymin, ymax)
    X_test_dm = reconstruct(X_test_dm, dmmin, dmmax)
    y_pred_dm = reconstruct(y_pred_dm, ymin, ymax)
    for arr in [X_test, y_true, y_pred, X_test_dm, y_pred_dm]:
        edge = arr[0,0,0]
        arr[arr==edge] = np.nan
    mdm = np.nansum(np.nansum(X_test_dm, axis=1), axis=1) * u.g.to('Msun')
    mfp = np.nansum(np.nansum(X_test, axis=1), axis=1) * u.g.to('Msun')
    
    lx_true = np.nansum(np.nansum(y_true, axis=1), axis=1) * dA 
    lx_pred = np.nansum(np.nansum(y_pred, axis=1), axis=1) * dA
    lx_pred_dm = np.nansum(np.nansum(y_pred_dm, axis=1), axis=1) * dA
    if type != 'lx':
        lx_true *= u.g.to('Msun') #Msun
        lx_pred *= u.g.to('Msun')
        lx_pred_dm *= u.g.to('Msun')
    if ret:
        return mdm, lx_pred_dm, mfp, lx_true, lx_pred
    if plot:
        fig, ax = plt.subplots()
        
        x = np.arange(13.7,15, .1)
        fit, cov = np.polyfit(np.log10(mfp), np.log10(lx_true), deg=1, cov=True)
        a, b = fit
        y = x*a + b
        ypred = np.log10(mfp)*a + b 
        err = np.log10(lx_true) - ypred
        std = np.std(ypred-err)
        plt.plot(10**x, 10**y, color='k')
        plt.fill_between(10**x, 10**(y - std), 10**(y + std), color='k', alpha=0.2)
        plt.scatter(mfp, lx_true, color='k', alpha=0.5, s=5, label='FP true')

        fit, cov = np.polyfit(np.log10(mfp), np.log10(lx_pred[:,0]), deg=1, cov=True)
        a, b = fit
        y = x*a + b
        ypred = np.log10(mfp)*a + b 
        err = np.log10(lx_true) - ypred
        std = np.std(ypred-err)
        plt.plot(10**x, 10**y, color='tab:blue')
        plt.fill_between(10**x, 10**(y - std), 10**(y + std), color='tab:blue', alpha=0.2)
        plt.scatter(mfp, lx_pred, color='tab:blue', alpha=0.5, s=5, label = 'FP pred')

        fit, cov = np.polyfit(np.log10(mdm), np.log10(lx_pred_dm[:,0]), deg=1, cov=True)
        a, b = fit
        y = x*a + b
        ypred = np.log10(mfp)*a + b 
        err = np.log10(lx_true) - ypred
        std = np.std(ypred-err)
        plt.plot(10**x, 10**y, color='tab:orange')
        plt.fill_between(10**x, 10**(y - std), 10**(y + std), color='tab:orange', alpha=0.2)
        plt.scatter(mdm, lx_pred_dm, color='tab:orange', alpha=0.5, s=10, label = 'DM pred')

        plt.xlabel(r'$M_{DM}/M_\odot$')
        plt.xscale('log')
        plt.yscale('log')
        if type == 'lx':
            plt.ylim(1e42, 1e46)
            plt.ylabel(r'$L_X$ (erg/s)')
        else:
            plt.ylim(1e13, 2e14)
            plt.ylabel(r'$M_g/M_\odot$')
        plt.xlim(6e13, 8e14)
        plt.legend()
        return fig, ax

def fit_scatter(mdm, lx):
    y = np.log10(lx) 
    x = np.log10(mdm) 
    fit = np.polyfit(x, y, 1, cov=True)
    return fit

def pred_error():
    mdm, lx_pred_dm, mfp, lx_true, lx_pred = scaling_relations(ret=True, plot=False)
    fit = fit_scatter(mfp, lx_true)
    a, b = fit[0]
    lx_fit = 10**(np.log10(mdm)*a + b)
    err = (lx_pred_dm[:,0] - lx_fit)/lx_fit
    return err

from astropy.io import fits 
from scipy.stats import spearmanr

def err_vs_cluster_props(modelname, prop1 = 'Group_M_Crit200', prop2 = 'GroupBHMass', errtype = 'mse'):
    #GroupBHMdot, GroupSFR
    norm = modelname.split('loss_')[1].split('-norm')[0]
    cat = pd.read_csv('groupcat-fp.csv')
    df = pd.read_csv('../model-normalization.csv')
    X = np.zeros((len(cat)*3, 512, 512))
    x = np.zeros(len(X))
    y = np.zeros(len(X))
    for i in range(len(cat)):
        filename = 'dm_mass_snap%d_halo%d_x.fits' % (int(cat['snap'][i]), int(cat['halo'][i]))
        try:
            X[i*3] = fits.getdata(filename)
            X[i*3 + 1] = fits.getdata(filename.replace('x','y'))
            X[i*3 + 2] = fits.getdata(filename.replace('x','z'))
            x[i*3: (i*3)+3] = cat[prop1][i]
            y[i*3: (i*3)+3] = cat[prop2][i]
        except:
            print(cat['snap'][i], cat['halo'][i], 'fail')

    lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df[norm]
    Xnorm = renorm(X, dmmin, dmmax)
    Xnorm[np.isnan(Xnorm)] = 0
    model = models.load_model(modelname)
    ypred = model.predict([Xnorm, (Xnorm > 0).astype(int)])
    prop = modelname.split('_in_')[1].split('_out')[0]
    if prop == 'lx':
        ymin, ymax = lxmin, lxmax
        ybase = 'lx_05_20_keV'
    if prop == 'rho':
        ymin, ymax = rhomin, rhomax
        ybase = 'gas_rho'
    if prop =='temp':
        ymin, ymax = tmin, tmax
        ybase = 'gas_temp'
    
    ypred = reconstruct(ypred[:,:,:,0], ymin, ymax)

    ytrue = np.zeros(X.shape)
    for i in range(len(cat)):
        filename = ybase+'_snap%d_halo%d_x.fits' % (int(cat['snap'][i]), int(cat['halo'][i]))
        try:
            ytrue[i*3] = fits.getdata(filename)
            ytrue[i*3 + 1] = fits.getdata(filename.replace('_x','_y'))
            ytrue[i*3 + 2] = fits.getdata(filename.replace('_x','_z'))
        except:
            print(cat['snap'][i], cat['halo'][i], 'fail')
    ytrue [ytrue ==0] = np.nan
    err = mse(ytrue, ypred)

    #groupcat must include only the test set
    #i guess it could also include validation set? Ask Michelle
    abserr = abserr [x>0]
    y = y [x>0]
    x = x [x>0]
    xlim = np.log10(x.min()), np.log10(x.max())
    ylim = np.log10(y.min()), np.log10(y.max())
    zlim = abserr.min(), abserr.max()
    spr_1 = spearmanr(abserr, x)
    spr_2 = spearmanr(abserr, y)
    print('Spearman R between error and %s : ' % prop1, spr_1)
    print('Spearman R between error and %s : ' % prop2, spr_2)

    fig, ax = plt.subplots()

    hb = ax.hexbin(np.log10(x), np.log10(y), C = abserr, gridsize=50, cmap='inferno', vmin=-0.1, vmax = 0.5)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_title("Median error vs cluster properties")
    cb = fig.colorbar(hb, ax=ax, label=r'$\Delta$/True')
    ax.set_ylabel(prop2)
    ax.set_xlabel(prop1)
    return fig, ax 

def mse(yt, yp):
    tsum = np.nansum(np.nansum(yt, axis=-1), axis=-1)
    psum = np.nansum(np.nansum(yp, axis=-1), axis=-1)
    err = np.sqrt(((tsum-psum))**2)
    return err #/len(tsum)

def mpe(yt, yp):
    tsum = np.nansum(np.nansum(yt, axis=-1), axis=-1)
    psum = np.nansum(np.nansum(yp, axis=-1), axis=-1)
    err =  (psum-tsum)/tsum 
    return err#/len(tsum)

def mape(yt, yp):
    tsum = np.nansum(np.nansum(yt, axis=-1), axis=-1)
    psum = np.nansum(np.nansum(yp, axis=-1), axis=-1)
    err =  abs((psum-tsum)/tsum)
    return err#len(tsum)

def plot_errors(modelnames):
    # fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex=False, sharey=False)
    for modelname in modelnames:
        norm = modelname.split('loss_')[1].split('-norm')[0]
        prop = modelname.split('_in_')[1].split('_out')[0]
        print(prop, norm)
        lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df[norm]
        if prop == 'lx':
            ymin, ymax = lxmin, lxmax
        if prop == 'rho':
            ymin, ymax = rhomin, rhomax
        if 'temp' in prop:
            ymin, ymax = tmin, tmax
        X_test, y_test, y_pred = extract_model(modelname)
        y_pred = y_pred[:,:,:,0]
        # err_mse = mse(y_test, y_pred) * 100  #to %
        # err_mpe = mpe(y_test, y_pred) * 100
        # err_mape = mape(y_test, y_pred) * 100
        # for err in [err_mpe, err_mape]:
        #     err[np.isinf(err)] = np.nan
        # ax[0][0].hist(err_mpe, bins=100, histtype='step', label=prop)
        # ax[0][1].hist(err_mape, bins=100, histtype='step', label=prop)
        if 'nolog' in modelname:
            ymin = 10**ymin
            ymax = 10**ymax
        y_test = reconstruct(y_test, ymin, ymax)
        y_pred = reconstruct(y_pred, ymin, ymax)
        print('reconstruct done')
        err_mse = mse(y_test, y_pred) * 100 #to %
        err_mpe = mpe(y_test, y_pred) * 100
        err_mape = mape(y_test, y_pred) * 100
        for err in [err_mse, err_mpe, err_mape]:#, err_mse]: 
            err[np.isinf(err)] = np.nan
            emax = np.nanpercentile(err, 90)
            err[err > emax] = emax
        # ax[1][0].hist(err_mpe, bins=100, histtype='step')
        # ax[1][1].hist(err_mape, bins=100,  histtype='step')
        # print(modelname, 'mean MSE: ', np.nanmean(err_mse), 'median MSE: ',np.nanmedian(err_mse))
        print(modelname, 'mean MPE: ', np.nanmean(err_mpe), 'median MPE: ',np.nanmedian(err_mpe))
        print(modelname, 'mean MAPE: ', np.nanmean(err_mape), 'median MAPE: ',np.nanmedian(err_mape))

def radial_profile(image):
    lenx = len(image)
    center = lenx/2
    X, Y = np.meshgrid(np.arange(lenx), np.arange(lenx))
    X = X.astype(float) - center
    Y = Y.astype(float) - center
    dx = 4000/512. #kpc
    r = np.sqrt(X**2 + Y**2) * dx 
    rbin = np.arange(0, 1001, 10) #in kpc 
    profile = np.zeros(len(rbin) - 1)
    for i in range(len(profile)):
        rmin = rbin[i]
        rmax = rbin[1+i]
        mask = (r > rmin)*(r < rmax)
        profile[i] = np.nanmean(image[mask])
    return profile

def compare_profiles(modelname, interquartile=False):
    X_test, y_test, y_pred = extract_model(modelname)
    prop = modelname.split('in_')[1].split('_out')[0]
    norm = modelname.split('loss_')[1].split('-norm')[0]
    lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df[norm]
    if prop == 'lx':
        ymin, ymax = lxmin, lxmax
    if prop == 'rho':
        ymin, ymax = rhomin, rhomax
    if 'temp' in prop:
        ymin, ymax = tmin, tmax
    y_test = reconstruct(y_test, ymin, ymax)
    y_pred = reconstruct(y_pred, ymin, ymax)
    X_test = reconstruct(X_test, dmmin, dmmax)
    rbins = np.arange(0, 1001, 10)
    profile_true = np.zeros((len(y_test), len(rbins)-1))
    profile_pred = np.zeros((len(y_test), len(rbins)-1))
    for i in range(len(y_test)):
        profile_true[i] = radial_profile(y_test[i])
        profile_pred[i] = radial_profile(y_pred[i])
    median_true = np.zeros(profile_true.shape[1])
    min_true = np.zeros(profile_true.shape[1])
    max_true = np.zeros(profile_true.shape[1])
    median_pred = np.zeros(profile_true.shape[1])
    min_pred = np.zeros(profile_true.shape[1])
    max_pred = np.zeros(profile_true.shape[1])
    
    for i in range(len(median_true)):
            median_true = np.nanmedian(profile_true, axis=0)
            median_pred = np.nanmedian(profile_pred, axis=0)
            if interquartile:
                min_true = np.nanpercentile(profile_true, 25, axis=0)
                max_true = np.nanpercentile(profile_true, 75, axis=0)
                min_pred = np.nanpercentile(profile_pred, 25, axis=0)
                max_pred = np.nanpercentile(profile_pred, 75, axis=0)
            else:
                min_true = np.nanmin(profile_true, axis=0)
                max_true = np.nanmax(profile_true, axis=0)
                min_pred = np.nanmin(profile_pred, axis=0)
                max_pred = np.nanmax(profile_pred, axis=0)
    return median_true, min_true, max_true, median_pred, min_pred, max_pred

def plot_profile(modelname, ax, interquartile = False, ymin=1e-8, ymax=0.1, ylabel=r'$\Sigma_X$ (erg/cm$^2$/s)'):
    rbins = np.arange(0, 1001, 10)
    median_true, min_true, max_true, median_pred, min_pred, max_pred = compare_profiles(modelname, interquartile=interquartile)    
    ax.plot(rbins[:-1], median_true, color='k', label='True')
    ax.plot(rbins[:-1], median_pred, color='tab:blue', label='Pred')
    ax.fill_between(rbins[:-1], min_true, max_true, color='k',alpha=0.1)
    ax.fill_between(rbins[:-1], min_pred, max_pred, color='tab:blue',alpha=0.1)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin, ymax)
    plt.xlim(1, 1000)
    plt.xscale('log')
    if 'temp' in modelname:
        if not interquartile:
            ax.set_title('Median + min-max range')
        else:
            ax.set_title('Median + interquartile range')
    else:
        ax.set_yscale('log')
    return fig, ax

def plot_all_profiles():
    fig, ax = plt.subplots(nrows = 2, ncols = 3, sharex=True, sharey=False)
    modelnames = ['mass_in_rho_out_mse-loss_4sigma-norm-mask.h5',
                'mass_in_temp_out_mse-loss_minmax-norm-mask.h5',
                'mass_in_lx_out_mse-loss_minmax-norm-mask.h5']
    plot_profile(modelnames[0], ax[0,0], ymin=1e-5, ymax=1, ylabel=r'$N_g$ (g/cm$^2$)')
    plot_profile(modelnames[0], ax[1,0], interquartile=True, ymin=1e-5, ymax=1, ylabel=r'$N_g$ (g/cm$^2$)')
    plot_profile(modelnames[1], ax[0,1], ymin=1, ymax=12, ylabel=r'$T_X$ (keV)')
    plot_profile(modelnames[1], ax[1,1], interquartile=True, ymin=1, ymax=12, ylabel=r'$T_X$ (keV)')
    plot_profile(modelnames[2], ax[0,2], ymin=1e-8, ymax=1, ylabel=r'$\Sigma_X$ (erg/cm$^2$/s)')
    plot_profile(modelnames[2], ax[1,2], interquartile=True, ymin=1e-8, ymax=1, ylabel=r'$\Sigma_X$ (erg/cm$^2$/s)')
    plt.legend()
    plt.tight_layout()
    fig.savefig('profiles.png', dpi=192)
    return fig, ax