import glob, os
import numpy as np
from astropy.io import fits
import pandas as pd

matches = np.load('fp_to_dm.npy')
workDir = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/highres/'
os.chdir(workDir)

#if cutouts_via_api:
base_path = None 
simname = 'TNG300-1'
#else specify the directory where /output contains the simulation snapshots

for snap in np.unique(matches[:,0]):
    sub = matches[matches[:,0]==snap]
    hmax = int(sub[-1, 1])
    for halo in range(hmax):
        if halo < 10:
            h = '00'+str(halo)
        elif halo < 100:
            h = '0'+str(halo)
        else:
            h = str(halo)
        if not glob.glob('gas_rho_snap%d_halo%s_z.fits' % (snap, h)):
            print(dir, snap, halo)
            # try:
            if not glob.glob('snap%d_halo%d_fp.hdf5' % (snap, halo)):
                if base_path:
                    save_halo_cutouts(simname, snap, halo, outname='snap%d_halo%d_fp.hdf5' % (snap, halo))
                else:
                    cutout(base_path, fields, snap, halo_id)
            try:
                yt_xray(snap, halo,'cutouts/'+dir,lx=False,dm=False,rho=True,temp=False)
            except:
                print('error')
            os.remove('snap%d_halo%d_fp.hdf5' % (snap, halo))

def compile_training_data(dir):
   os.chdir()
   fdm = glob.glob('dm_mass_*fits'); fdm.sort() 
   fsb = [d.replace('dm_mass','lx_05_20_keV') for d in fdm]
   ftemp = [d.replace('dm_mass','gas_temp') for d in fdm]
   frho = [d.replace('dm_mass','gas_rho') for d in fdm]
   dm = np.zeros((len(dm), 512, 512, 1))
   rho = np.zeros((len(dm), 512, 512, 1))
   temp = np.zeros((len(dm), 512, 512, 1))
   sb = np.zeros((len(dm), 512, 512, 1))
   for i in range(len(inputs)):
       dm[i, :,:,0]   = fits.getdata(fdm[i])
       rho[i, :,:,0]   = fits.getdata(frho[i])
       temp[i, :,:,0] = fits.getdata(ftemp[i])
       sb[i, :,:,0]   = fits.getdata(fsb[i])
   np.save('dm.npy', dm)
   np.save('rho.npy', rho)
   np.save('temp.npy', temp)
   np.save('lx.npy', sb)
    
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
    ret /= (arrmax-arrmin+eps) 
    return ret #to reconstruct arr, need np.nanmin(arr), np.nanmax(arr)

def norm_limit(arr, nsigma=5):
    mean = np.log10(arr[arr>0]).mean()
    std = np.log10(arr[arr>0]).std()
    return mean - nsigma*std, mean + nsigma*std

def save_normalisations():
   names = []
   min = []
   max = []
   minsigma = []
   maxsigma = []
   for file in ['lx.npy', 'rho.npy', 'temp.npy', 'dm.npy']:
      names.append(file.split('.')[0])
      arr = np.load(file)
      smin, smax = norm_limit(arr, 4)
      minsigma.append(smin)
      maxsigma.append(smax)
      arr = np.log10(arr[arr > 0])
      min.append(np.nanmin(arr))
      max.append(np.nanmax(arr))
   df = pd.DataFrame()
   df.insert(0, 'minmax', np.zeros(8)]
   df.insert(1, '4sigma', np.zeros(8)]
   df['minmax'][::2]  = min
   df['minmax'][1::2] = max
   df['4sigma'][::2]  = smin
   df['4sigma'][1::2] = smax
   df.to_csv('model-normalizations.csv')

for norm in ['minmax', '4sigma']:
   lxmin, lxmax, rhomin, rhomax, ktmin, ktmax, dmmin, dmmax = df[norm] 
   dm = np.load('dm.npy')
   rho = np.load('rho.npy')
   temp = np.load('temp.npy')
   lx = np.load('lx.npy')
   dnorm = renorm(dm, dmmin, dmmax)
   rnorm = renorm(rho, rhomin, rhomax)
   tnorm = renorm(temp, ktmin, ktmax)
   lnorm = renorm(lx, lxmin, lxmax)
   np.save('dm-normed-%s.npy' % norm, dnorm)
   np.save('rho-normed-%s.npy' % norm, rnorm)
   np.save('temp-normed-%s.npy' % norm, tnorm)
   np.save('lx-normed-%s.npy' % norm, lnorm)
