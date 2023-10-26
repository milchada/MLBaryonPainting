import glob
import numpy as np
from astropy.io import fits

def compile_training_data():
   fdm = glob.glob('dm_Mass_*npy'); dm.sort() 
   fgas = [d.replace('dm_Mass','gas_Density') for d in dm]
   ftemp = [d.replace('dm_Mass','gas_Temperature') for d in dm]
   fbh = [d.replace('dm_Mass','bh_Mdot') for d in dm]
   dm = np.zeros((len(dm), 512, 512, 1))
   bh = np.zeros((len(dm), 512, 512, 1))
   temp = np.zeros((len(dm), 512, 512, 1))
   gas = np.zeros((len(dm), 512, 512, 1))
   for i in range(len(inputs)):
       dm[i, :,:,0] = np.load(fdm[i])
       bh[i, :,:,0] = np.load(fbh[i])
       temp[i, :,:,0] = np.load(ftemp[i])
       gas[i, :,:,0] = np.load(fgas[i])
   np.save('dm.npy', dm)
   np.save('bh.npy', bh)
   np.save('temp.npy', temp)
   np.save('rho.npy', rho)
    
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
