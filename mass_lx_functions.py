import numpy as np
import matplotlib.pylab as plt
from astropy import units as u 
import os, gc
from keras import models
import tensorflow as tf

# basePath = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/'
dx = (4000./512) * u.kpc.to('cm') #kpc/pixel --> cm/pix; lx is in photons/cm**2/s
dA = dx**2
h = 0.6774 
fgas = 0.17 #Omega_b/Omega_m

def norm_limit(arr, nsigma=5):
    mean = np.log10(arr[arr>0]).mean()
    std = np.log10(arr[arr>0]).std()
    return mean - nsigma*std, mean + nsigma*std

limits = {'lx' : (43, 46),'rho' : (13,14.5),'dm' : (14,15.3),'norm':{0,1}}

def dN_dVdlogX(X, nbins=100, limits = None, log=False, nsnaps=4, ftest = 0.1): 
		#nsnaps determines total volume
		#ftest is the fraction of clusters used in the histogram; so must scale UP by inverse
	if log:
		logX = np.log10(X)
	else:
		logX = X
	if not limits:
		xmin, xmax = (logX[X > 0].min(), logX.max())
	else:
		xmin, xmax = limits
	hist, bins = np.histogram(logX, range = (xmin, xmax), bins=nbins)
	V = ((300*u.Mpc/h)**3) * nsnaps
	dX = bins[1] - bins[0] #check that this is uniform
	phi = np.zeros(hist.shape)
	for i in range(len(phi)):
		phi[i] = np.sum(hist[i:])
	err = np.sqrt(phi)/phi
	return phi/(V.value * dX * ftest), (bins[:-1] + bins[1:])/2., err

def compute_dir(prop, norm='minmax', mask='', dmo = False, ftest=0.1):
	modelname = 'mass_in_%s_out_mse-loss_%s-norm%s.h5' % (prop, norm, mask)
	if norm == '4sigma':
	    lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df['4sigma'].values
	else:
	    lxmin, lxmax, rhomin, rhomax, tmin, tmax, dmmin, dmmax = df['minmax'].values
	if prop == 'lx':
		ymin, ymax = lxmin, lxmax
	elif prop =='rho':
		ymin, ymax = rhomin, rhomax
	else:
		ymin, ymax = tmin, tmax
	
	if dmo:
		X = np.load('../%s/dm-normed-%s.npy' % (dmo, norm))
		seed = np.random.RandomState(42) #this way it'll always be the same clusters 
		inds = seed.randint(0, len(X), size=len(X))
		test = inds[round(len(X)*.9):]
		X_test = X[test]
		del(X); gc.collect()
		X_test[np.isnan(X_test)] = 0
		model = models.load_model(modelname, compile=False)
		model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
		X_mask = (X_test > 0).astype(int)
		y_pred = model.predict([X_test, X_mask])
	else:
		X_test, y_test, y_pred = extract_model(modelname)
		
	if not dmo:
		y_test = reconstruct(y_test, ymin, ymax)
	y_pred = reconstruct(y_pred, ymin, ymax)
	X_test = reconstruct(X_test, dmmin, dmmax)
	if prop == 'lx':
		if not dmo:
			y_test = np.nansum(np.nansum(y_test, axis=1), axis=1) * dA
		y_pred = np.nansum(np.nansum(y_pred, axis=1), axis=1) * dA
	if prop == 'rho':
		if not dmo:
			y_test = np.nansum(np.nansum(y_test, axis=1), axis=1) * dA * u.g.to('Msun')
		y_pred = np.nansum(np.nansum(y_pred, axis=1), axis=1) * dA * u.g.to('Msun')
	
	if not dmo:
		true_function, true_bins, true_err = dN_dVdlogX(y_test, limits=limits[prop], log=True, ftest = ftest)
	pred_function, pred_bins, pred_err = dN_dVdlogX(y_pred, limits=limits[prop], log=True, ftest = ftest)
	
	mdm = np.nansum(np.nansum(X_test, axis=1), axis=1) * u.g.to('Msun') #this is units of g 

	mdm_function, mdbins, merr = dN_dVdlogX(mdm,limits=limits['dm'], log=True, ftest = ftest)

	if not dmo:
		return true_function, true_bins, true_err, pred_function, pred_bins, pred_err, mdm_function, mdbins, merr
	else:
		return pred_function, pred_bins, pred_err, mdm_function, mdbins, merr

# from scipy.interpolate import CubicSpline as spline
# def smooth(bins, mean, std, npts = 300):
# 	valid = (~np.isnan(std))*(~np.isnan(mean))
# 	xnew = np.linspace(bins.min(), bins.max(), npts)
# 	mean_smooth = spline(bins[valid], mean[~np.isnan(std)])(xnew)
# 	min_smooth = spline(bins[valid], (mean*(1-std))[valid])(xnew)
# 	max_smooth = spline(bins[valid], (mean*(1+std))[valid])(xnew)
# 	return xnew, mean_smooth, min_smooth, max_smooth

def allprops():
	fig, ax = plt.subplots(ncols = 2, sharex=False, sharey=True, figsize=(6,3))
	props = ['lx', 'rho']
	colors = {'minmax':'tab:blue','minmax-mask':'tab:green', '4sigma-mask':'tab:orange'}
	lines = {'minmax':'dashed','minmax-mask':'dashdot', '4sigma-mask':'dotted'}

	props = [['lx', 'minmax', ''],  ['lx', 'minmax', '-mask'],  ['lx', '4sigma', '-mask'], 
			 ['rho', 'minmax', ''], ['rho', 'minmax', '-mask'], ['rho', '4sigma', '-mask']]
	ind = {'lx':0, 'rho': 1}
	for prop in props:
		norm = prop[1]
		mask = prop[2]
		true_function, true_bins, true_err, pred_function, pred_bins, pred_err, mdm_function, mdbins, merr = compute_dir(prop[0], norm=prop[1], mask=prop[2])
		# tnew, tmean, tmin, tmax = smooth(true_bins, true_function, true_err)
		# pnew, pmean, pmin, pmax = smooth(pred_bins, pred_function, pred_err)
		i = ind[prop[0]]
		if (prop[1] == 'minmax' ) and (prop[2] == ''):
			ax[i].plot(true_bins, true_function, color='k', label='True')
			ax[i].fill_between(true_bins, true_function*(1-true_err), true_function*(1+true_err), color='k', alpha=0.25)
		ax[i].plot(pred_bins, pred_function, color=colors[norm+mask], label=norm+mask, linestyle=lines[norm+mask])
		ax[i].fill_between(pred_bins, pred_function*(1-pred_err), pred_function*(1+pred_err), color=colors[norm+mask], alpha=0.25)

	plt.yscale('log')
	h, l = ax[1].get_legend_handles_labels()
	plt.legend(loc='best')
	ax[0].set_ylabel(r'$\phi(L_X)$')
	ax[1].set_ylabel(r'$\phi(M_g)$')
	ax[0].set_xlabel(r'$L_X$ (erg/s)')
	ax[1].set_xlabel(r'$M_g (M_\odot)$')
	plt.ylim(3e-7,6e-4)
	ax[0].set_xlim(43,46.1)
	ax[1].set_xlim(13,14.5)
	plt.tight_layout()
	plt.savefig('/n/home07/uchadaya/functions.png', dpi=192)
	# return fig, ax


def fp_vs_dm():
	fig, ax = plt.subplots(ncols = 2, sharex=False, sharey=True, figsize=(6,3))
	props = [['lx', 'minmax'], ['rho', '4sigma']]

	for prop in props:
		i = props.index(prop)
		true_function, true_bins, true_err, pred_function, pred_bins, pred_err, mdm_function, mdbins, merr = compute_dir(prop[0], norm=prop[1], mask='-mask')
		true_err[np.isnan(true_err)] = 0
		pred_err[np.isnan(pred_err)] = 0
		ax[i].plot(true_bins, true_function, color='k', label='True')
		ax[i].fill_between(true_bins, true_function*(1-true_err), true_function*(1+true_err), color='k', alpha=0.25)
		ax[i].plot(pred_bins, pred_function, color='tab:blue', label=r'$M_{DM}$ from FP', linestyle='dashed')
		ax[i].fill_between(pred_bins, pred_function*(1-pred_err), pred_function*(1+pred_err), color='tab:blue', alpha=0.25)
		
		pred_function, pred_bins, pred_err, mdm_function, mdbins, merr = compute_dir(prop[0], norm=prop[1], mask='-mask', dmo='dmo')
		pred_err[np.isnan(pred_err)] = 0
		color = 'tab:orange'
		ax[i].plot(pred_bins, pred_function, color=, label=r'$M_{DM}$ from DMO', linestyle='dotted')
		ax[i].fill_between(pred_bins, pred_function*(1-pred_err), pred_function*(1+pred_err), color=color, alpha=0.25)
	
	plt.yscale('log')
	plt.legend()
	ax[0].set_ylabel(r'$\phi(L_X)$')
	ax[1].set_ylabel(r'$\phi(M_g)$')
	ax[0].set_xlabel(r'$L_X$ (erg/s)')
	ax[1].set_xlabel(r'$M_g (M_\odot)$')
	plt.ylim(3e-7,6e-4)
	plt.tight_layout()
	ax[0].set_xlim(43,46.1)
	ax[1].set_xlim(13,14.5)
	plt.savefig('/n/home07/uchadaya/functions_dmo.png', dpi=192)
	# return fig, ax

def res_test():
	fig, ax = plt.subplots(ncols = 2, sharex=False, sharey=True, figsize=(6,3))
	props = [['lx', 'minmax'], ['rho', '4sigma']]

	for prop in props:
		i = props.index(prop)
		true_function, true_bins, true_err, pred_function, pred_bins, pred_err, mdm_function, mdbins, merr = compute_dir(prop[0], norm=prop[1], mask='-mask')
		true_err[np.isnan(true_err)] = 0
		pred_err[np.isnan(pred_err)] = 0
		ax[i].plot(true_bins, true_function, color='k', label='True')
		ax[i].fill_between(true_bins, true_function*(1-true_err), true_function*(1+true_err), color='k', alpha=0.25)
		ax[i].plot(pred_bins, pred_function, color='tab:blue', label=r'$M_{DM}$ from FP-1', linestyle='dashed')
		ax[i].fill_between(pred_bins, pred_function*(1-pred_err), pred_function*(1+pred_err), color='tab:blue', alpha=0.25)
		
		pred_function, pred_bins, pred_err, mdm_function, mdbins, merr = compute_dir(prop[0], norm=prop[1], mask='-mask', dmo='midres')
		pred_err[np.isnan(pred_err)] = 0
		ax[i].plot(pred_bins, pred_function, color='tab:green', label=r'$M_{DM}$ from FP-2', linestyle='dashdot')
		ax[i].fill_between(pred_bins, pred_function*(1-pred_err), pred_function*(1+pred_err), color='tab:green', alpha=0.25)

		pred_function, pred_bins, pred_err, mdm_function, mdbins, merr = compute_dir(prop[0], norm=prop[1], mask='-mask', dmo='lowres')
		pred_err[np.isnan(pred_err)] = 0
		ax[i].plot(pred_bins, pred_function, color='tab:orange', label=r'$M_{DM}$ from FP-3', linestyle='dotted')
		ax[i].fill_between(pred_bins, pred_function*(1-pred_err), pred_function*(1+pred_err), color='tab:orange', alpha=0.25)
	
	plt.yscale('log')
	plt.legend()
	ax[0].set_ylabel(r'$\phi(L_X)$')
	ax[1].set_ylabel(r'$\phi(M_g)$')
	ax[0].set_xlabel(r'$L_X$ (erg/s)')
	ax[1].set_xlabel(r'$M_g (M_\odot)$')
	plt.ylim(3e-7,6e-4)
	ax[0].set_xlim(43,46.1)
	ax[1].set_xlim(13,14.5)
	plt.tight_layout()
	plt.savefig('/n/home07/uchadaya/functions-restest.png', dpi=192)
	# return fig, ax


def inputs():
	dm_4sigma = np.load('dm-normed-4sigma.npy')
	dmo_4sigma = np.load('../dmo/dm-normed-4sigma.npy')
	dm_minmax = np.load('dm-normed-minmax.npy')
	dmo_minmax = np.load('../dmo/dm-normed-minmax.npy')
	fig, ax = plt.subplots(ncols=2, sharey=True)
	hist1, bins1 = np.histogram(dm_4sigma[dm_4sigma>-100], bins=100)
	hist2, bins2 = np.histogram(dmo_4sigma[dmo_4sigma>-100], bins=100)
	hist3, bins3 = np.histogram(dm_minmax[dm_minmax>-100], bins=100)
	hist4, bins4 = np.histogram(dmo_minmax[dmo_minmax>-100], bins=100)
	ax[0].step(bins1[1:], hist1/len(dm_4sigma), label='FP')
	ax[0].step(bins2[1:], hist2/len(dmo_4sigma), label='DMO')
	ax[1].step(bins3[1:], hist3/len(dm_4sigma), label='FP')
	ax[1].step(bins4[1:], hist4/len(dmo_4sigma), label='DMO')
	ax[0].set_title(r'4-$\sigma$')
	ax[1].set_title('min-max')
	plt.legend()
	return fig, ax

