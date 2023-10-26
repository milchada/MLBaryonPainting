import yt, gc,os, glob
import numpy as np
import illustris_python as il 

# simdir = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG'
simdir = '/n/holystore01/LABS/hernquist_lab/Lab/IllustrisTNG/Runs/L205n2500TNG_DM'
matchdir = simdir+'/postprocessing/SubhaloMatchingToDark/'
writedir = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPainting/TNG-training-data/'

#step 1 - find halos that are > 1e14 Msun
basePath = simdir+'/output/'
h = 0.6774 

def make_catalog(filename):
   tot = 0
   k = 0
   match = np.zeros((13000,3))
   for snap in [ 67, 78, 99]: #range(50,100) 50,
      m200 = il.groupcat.loadHalos(basePath,snap,fields=['Group_M_Crit200'])*1e10/h #Msun
      select = np.argwhere(m200>1e14)[:,0]
      tot+=len(select)
      print(snap)
      matchfile = il.sublink.h5py.File(matchdir+'LHaloTree_0%d.hdf5' % snap)
      fp_id = np.zeros(select.max()+1)
      dm_id = np.zeros(select.max()+1)
      for i in range(len(fp_id)):
         fp_id[i] = matchfile['SubhaloIndexFrom'][i]
         dm_id[i] = matchfile['SubhaloIndexTo'][i]
      _, valid, _ = np.intersect1d(fpid.astype(int),select, return_indices=True)

      match[k:k+len(valid), 0] = snap
      match[k:k+len(valid), 1] = fp_id[valid] 
      match[k:k+len(valid), 2] = dm_id[valid]
      k+=len(valid)
      print('DM matches logged')

   match = match[:tot+1]
   np.save(filename, match)
