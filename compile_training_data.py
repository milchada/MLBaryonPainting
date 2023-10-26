import glob
import numpy as np
from astropy.io import fits

matches = np.load('fp_to_dm.npy')
simname = 'TNG300-1'
dir = 'highres/'

#if cutouts_via_api:
os.chdir('/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/'+dir)
for snap in np.unique(matches[:,0]):
    sub = matches[matches[:,0]==snap]
    hmax = int(sub[-1, 1])
    for halo in range(338,339):#hmax):
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
                save_halo_cutouts(simname, snap, halo, outname='snap%d_halo%d_fp.hdf5' % (snap, halo))
            try:
                yt_xray(snap, halo,'cutouts/'+dir,lx=False,dm=False,rho=True,temp=False)
            except:
                print('error')
            os.remove('snap%d_halo%d_fp.hdf5' % (snap, halo))

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
