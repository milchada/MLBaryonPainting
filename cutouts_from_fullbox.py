import numpy as np
import illustris_python as il
import gc 

import pandas as pd
groupcat = pd.read_csv('groupcat.csv')

# loc = Path("/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/")
# loc = Path("/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n1250TNG/") #midres
# loc = Path("/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n625TNG/") #lowres
fields = {
    # 0: ["Coordinates", "Velocities", "Masses", "Density", "ElectronAbundance",
        # "NeutralHydrogenAbundance", "InternalEnergy", "GFM_Metallicity", "GFM_Metals",
        # "ParticleIDs", "StarFormationRate"],
    1: ["Coordinates", "Velocities", "ParticleIDs"]#,
    # 4: ["Coordinates", "Velocities", "ParticleIDs", "Masses"]
}
base_path = str(loc /"output")
def cutout(bar_path, fields, snap, halo_id):
    if 'DM' in loc.name:
        ptypes = [1]
    else:
        ptypes = [0, 1, 4]
    filename = "output/snapdir_0%d/snap_0%d.%d.hdf5" % (snap,snap, halo_id)
    with h5py.File(loc/filename, "r") as s:
        with h5py.File(f"snap{snap}_halo_{halo_id}.hdf5", "w") as f:
            s.copy("Header", f)
            s.copy("Parameters", f)
            s.copy("Config", f)
            h = f["Header"]
            pos = s['PartType1']['Coordinates']
            xyz = np.zeros(pos.shape)
            for i in range(3):
                xyz[:,i] = pos[:,i]
            left = xyz.min(axis=0)
            right = xyz.max(axis=0)
            num_parts = np.zeros(6, dtype='uint32')
            for ptype in ptypes:
                halo = il.snapshot.loadHalo(base_path, snap, halo_id, 
                                        ptype, fields=fields[ptype])
                num_parts[ptype] = halo["Coordinates"].shape[0]
                g = f.create_group(f"PartType{ptype}")
                for key in halo:
                    g.create_dataset(key, data=halo[key])
            h.attrs["NumPart_Total"] = num_parts
            h.attrs["NumPart_ThisFile"] = num_parts
            h.attrs["NumFilesPerSnapshot"] = 1
            f.close()


def projection(pos, weight, cmin, cmax, type, fieldname, snap,halo, nbins=512, prefix=''):   
   hist, _, _ = np.histogram2d(pos[:,1], pos[:,2], weights=weight, range=((cmin[1], cmax[1]),(cmin[2],cmax[2])),  bins=nbins)
   np.save('%s%s_%s_%d_%d_x.npy' % (prefix, type, fieldname, snap, halo), hist)
   print('x done')
   hist, _, _ = np.histogram2d(pos[:,2], pos[:,0], weights=weight, range=((cmin[2], cmax[2]),(cmin[0],cmax[0])),  bins=nbins)
   np.save('%s%s_%s_%d_%d_y.npy' % (prefix, type, fieldname,snap, halo), hist)
   del(hist); gc.collect()
   print('y done')
   hist, _, _ = np.histogram2d(pos[:,0], pos[:,1], weights=weight, range=((cmin[0], cmax[0]),(cmin[1],cmax[1])),  bins=nbins)
   np.save('%s%s_%s_%d_%d_z.npy' % (prefix, type, fieldname, snap, halo), hist)
   del(hist); gc.collect()
   print('z done')

def temp_K(Eint, xe):
   #reference: https://www.tng-project.org/data/docs/faq/#gen6
   gamma = 5./3
   X_H = 0.76
   from astropy.constants import k_B, m_p
   kB = k_B.to('keV/K').value
   mu = 4/(1 + 3*X_H + 4*X_H*xe) * m_p.to('g').value
   return (gamma - 1) * Eint * 1e10 * mu / kB  

def get_fp_cutouts(todo):
   basePath = '/n/holystore01/LABS/hernquist_lab/Lab/IllustrisTNG/Runs/L205n2500TNG/output/'
   for i in todo.index:
      snap = todo['snap'][i]
      halo = todo['halo'][i]
      dm = il.snapshot.loadHalo(basePath,int(snap), int(halo),'dm', fields=['Coordinates', 'Potential'])
      gas = il.snapshot.loadHalo(basePath,int(snap), int(halo),'gas', fields=['Coordinates','Density','InternalEnergy','ElectronAbundance'])
      # bh = il.snapshot.loadHalo(basePath,int(snap), int(halo),'bh', fields = ['Coordinates', 'BH_Mdot'])
      print("Loaded")
      pos = dm['Coordinates']
      c = pos[np.argmin(dm['Potential'])]
      cmin = c-4000
      cmax = c+4000      
      # projection(pos, weight=None, cmin=cmin, cmax=cmax, type='dm', fieldname='Mass', snap=snap,halo=halo, nbins=512)

      temp = temp_K(gas['InternalEnergy'], gas['ElectronAbundance'])
      projection(pos=gas['Coordinates'], weight=temp, cmin=cmin, cmax=cmax, type='gas', fieldname='Temperature', snap=snap,halo=halo, nbins=512)
      # projection(pos=gas['Coordinates'], weight=gas['Density'], cmin=cmin, cmax=cmax, type='gas', fieldname='Density', snap=snap,halo=halo, nbins=512)
      # projection(pos=bh['Coordinates'], weight=bh['BH_Mdot'], cmin=cmin, cmax=cmax, type='bh', fieldname='Mdot', snap=snap,halo=halo, nbins=512)

      # del(gas, dm, bh, temp, pos)
      del(dm, temp, pos)
      gc.collect()
      print(snap, halo, type, 'done')

def fp_dm_cutouts(match):
   fpPath = '/n/holystore01/LABS/hernquist_lab/Lab/IllustrisTNG/Runs/L205n2500TNG/output/'
   dmPath = '/n/holystore01/LABS/hernquist_lab/Lab/IllustrisTNG/Runs/L205n2500TNG_DM/output/'
   for row in match:
      snap, fpind, dmind = row.astype(int)
      fp = il.snapshot.loadHalo(fpPath, snap, fpind,'dm', fields=['Coordinates', 'Potential'])
      dm = il.snapshot.loadHalo(dmPath, snap, dmind,'dm', fields=['Coordinates', 'Potential'])
      pos = dm['Coordinates']
      c = pos[np.argmin(dm['Potential'])]
      cmin = c-4000
      cmax = c+4000
      projection(pos, weight=None, cmin=cmin, cmax=cmax, type='dm', fieldname='Mass', snap=snap,halo=fpind, nbins=512, prefix='dmo_')
      pos = fp['Coordinates']
      c = pos[np.argmin(fp['Potential'])]
      cmin = c-4000
      cmax = c+4000
      projection(pos, weight=None, cmin=cmin, cmax=cmax, type='dm', fieldname='Mass', snap=snap,halo=fpind, nbins=512, prefix='fp_')

def compile_training_data():
   import glob
   dm = glob.glob('dm_Mass_*npy'); dm.sort() 
   gas = [d.replace('dm_Mass','gas_Density') for d in dm]
   temp = [d.replace('dm_Mass','gas_Temperature') for d in dm]
   bh = [d.replace('dm_Mass','bh_Mdot') for d in dm]
   inputs = np.zeros((len(dm), 512, 512, 2))
   outputs = np.zeros((len(dm), 512, 512, 2))
   for i in range(len(inputs)):
       inputs[i, :,:,0] = np.load(dm[i])
       inputs[i, :,:,1] = np.load(bh[i])
       outputs[i, :,:,0] = np.load(gas[i])
       outputs[i, :,:,1] = np.load(bh[i])
   np.save('dm_bhar.npy', inputs)
   np.save('gasrho_temp.npy', outputs)

   def renorm(arr, eps=0.1, log=True):
       if log:
           arr = np.log10(arr)
           arr[arr == -np.inf] = np.nan
       arr -= np.nanmin(arr)
       arr += eps
       arr /= np.nanmax(arr)
       return arr

   inputs[:,:,:,0] = renorm(inputs[:,:,:,0])
   inputs[:,:,:,1] = renorm(inputs[:,:,:,1])
   outputs[:,:,:,0] = renorm(outputs[:,:,:,0])
   outputs[:,:,:,1] = renorm(outputs[:,:,:,1])
   gc.collect()
   inputs[np.isnan(inputs)] = 0
   outputs[np.isnan(outputs)] = 0
   np.save('inputs.npy', inputs)
   np.save('outputs.npy', outputs)

def get_group_props(matches, basePath = '/n/holystore01/LABS/hernquist_lab/Lab/IllustrisTNG/Runs/L205n2500TNG/output/'):
   from astropy import units as u
   import gc
   import pandas as pd 
   import numpy as np 

   if 'DM' in basePath:
      fields = ['Group_M_Crit200','GroupNsubs','Group_R_Crit200', 'GroupPos']
      fieldnames = ['Group_M_Crit200','GroupNsubs','Group_R_Crit200', 'GroupPos']
   else:
      fields = ['Group_M_Crit200', 'GroupBHMass', 'GroupBHMdot','GroupNsubs', 'GroupSFR', 'Group_R_Crit200', 'GroupPos']
      fieldnames = ['Group_M_Crit200', 'GroupBHMass', 'GroupBHMdot','GroupNsubs', 'GroupSFR', 'Group_R_Crit200', 'GroupPos'] 
   if 'GroupPos' in fields:
      groupcat = np.zeros((len(matches), len(fields)+4))
      fieldnames[-1] = 'GroupPosX'
      fieldnames += ['GroupPosY', 'GroupPosZ']
   else:
      groupcat = np.zeros((len(matches), len(fields)+2))
   i=0
   for snap in [50,67,78,99]:#np.unique(todo[:,0]):
      sub = matches[matches[:,0] == snap]
      if 'DM' in basePath:
         hnums = sub[:,2].astype(int)
      else:
         hnums = sub[:,1].astype(int)
      cat = il.groupcat.loadHalos(basePath,int(snap),fields=fields)
      groupcat[i:i+len(hnums), 0] = snap 
      groupcat[i:i+len(hnums), 1] = hnums
      for field in fields:
         try:
            groupcat[i:i+len(hnums), 2+fields.index(field)] = cat[field][hnums]
         except ValueError: #this will happen for GroupPos
            groupcat[i:i+len(hnums), 2+fields.index(field):5+fields.index(field)] = cat[field][hnums]
      i+=len(sub)
      del(cat)
      gc.collect()
      print(int(snap), 'done')
   
   df = pd.DataFrame(data = groupcat, columns=['snap', 'halo']+fieldnames)
   h =  il.groupcat.loadHeader(basePath, snapNum=int(snap))['HubbleParam']
   munit = 1e10*u.Msun/h
   if 'DM' not in basePath:
      df['GroupBHMdot'] *= 1e10*u.Msun/(0.978*u.Gyr)
      df['GroupBHMass'] *= munit
   df['Group_M_Crit200'] *= munit
   return df
