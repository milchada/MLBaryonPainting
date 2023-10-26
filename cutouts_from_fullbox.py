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

def make_proj(filename, snapnum, halonum, dm=True, gas=True, ns=["x","y","z"], suffix=''):
   ds = yt.load(filename)
   try:
      _, c = ds.find_min(("PartType5","Potential"))
   except:
      c = 'c'
   for n in ns:
      if gas:
         if not glob.glob('gas_sb_proj_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix)):
            xray_fields = yt.add_xray_emissivity_field(ds, 0.3, 7, table_type='apec', metallicity=0.3)
            p = yt.FITSProjection(ds, n, ("gas","xray_photon_emissivity_0.3_7_keV"), center=c, width=(8, "Mpc"))
            p.writeto('gas_sb_proj_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix), overwrite=True)
            del(p)
            gc.collect()
         if not glob.glob('gas_kT_proj_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix)):
            p = yt.FITSProjection(ds, n, ("gas","kT"), center=c, width=(8, "Mpc"), weight_field="mazzotta_weighting")
            p.writeto('gas_kT_proj_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix), overwrite=True)
            del(p)
            gc.collect()
         if not glob.glob('bh_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix)):
            p = yt.FITSParticleProjection(ds, n, ('PartType5', 'BH_Mdot'), center=c, width=(8, "Mpc"), deposition="cic")
            p.writeto('bh_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix), overwrite=True)
            del(p)
            gc.collect()
      if dm:
         if not glob.glob('dm_massproj_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix)):
            p = yt.FITSParticleProjection(ds, n, ("PartType1","particle_mass"), center=c, width=(8, "Mpc"), deposition="cic")
            p.writeto('dm_massproj_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix), overwrite=True)
            del(p)
            gc.collect()
         
   #now select only those that are valid in all fields, right?
   allmask = (np.sum(dmmass, axis = 1) > 0) * (np.sum(dmvel, axis = 1) > 0) * (np.sum(sb, axis = 1) > 0) 
            * (np.sum(kt, axis = 1) > 0) * (np.sum(bh, axis = 1) > 0)
   dmmass = dmmass[allmask]
   dmvel  = dmvel[allmask]
   sb     = sb[allmask]
   kt     = kt[allmask]
   bh     = bh[allmask]

   np.save('dmmass.npy', dmmass)
   np.save('dmvel.npy', dmvel)
   np.save('sb.npy', sb)
   np.save('kT.npy', kt)
   np.save('bhmdot.npy', bh)

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
