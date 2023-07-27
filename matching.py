import numpy as np
import illustris_python as il
from astropy import units as u

basePath = '/n/holystore01/LABS/hernquist_lab/Lab/IllustrisTNG/Runs/L205n2500TNG/output/'
matchdir = basePath.replace('output','postprocessing/SubhaloMatchingToDark')

def make_catalog(filename, nmax=1000, field='GroupMass'): #Group_M_Crit200
   k = 0
   match = np.zeros((nmax,3))
   for snap in [50,67,78,99]:#range(52, 100):
      h =  il.groupcat.loadHeader(basePath, snapNum=int(snap))['HubbleParam']
      munit = 1e10*u.Msun/h
      m200 = il.groupcat.loadHalos(basePath,snap,fields=field)*1e10/h #Msun
      select = np.argwhere(m200>1e14)[:,0]
      print("Masses collected")

      matchfile = il.sublink.h5py.File(matchdir+'LHaloTree_0%d.hdf5' % snap)
      fpind = matchfile['SubhaloIndexFrom'] 
      dmind = matchfile['SubhaloIndexTo']
      valid = fpind[fpind < select.max()]
      for ind in valid:
         match[k] = [snap, ind, dmind[fpind == ind]]
         k+=1
      print('DM matches logged')

   print(len(match[np.sum(match, axis=1) > 0]))
   match = match[np.sum(match, axis=1) > 0]
   np.save(filename, match)
