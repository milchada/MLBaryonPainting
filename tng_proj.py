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

def make_proj(fn, snapnum, halonum, dm=True, gas=True, ns=["x","y","z"], suffix=''):
   ds = yt.load(fn)
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
         # if not glob.glob('dm_velmag_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix)):
         #this won't work in 2D
         #    p = yt.FITSParticleProjection(ds, n, ("PartType1","particle_velocity_magnitude"), center=c, width=(8, "Mpc"), deposition="cic", weight_field='particle_mass')
         #    p.writeto('dm_velmag_%s_%s_%s%s.fits' % (snapnum, halonum, n, suffix), overwrite=True)
         #    del(p)
            # gc.collect()

def save_halo_cutouts(simname, snapnum, halonum, outname=None):    
   file = 'http://www.tng-project.org/api/%s/snapshots/%d/halos/%d/cutout.hdf5' % (simname, snapnum, halonum)
   halo = get(file)
   if outname:
      with open(outname, 'wb') as f:
         f.write(halo.content)
         del(halo, f)
         gc.collect()
   else:
      return halo

def pull_and_make_fits(smin, smax,match):
   for snap in range(smin,smax):
      mcut = match[match[:,0] == snap]
      _, ind = np.unique(mcut[:,2], return_index = True ) #because sometimes multiple FP halos matched to same DM halo
      mcut = mcut[ind]
      for pair in mcut:
         # if not glob.glob('gas_sb_proj_%d_%d_z.fits' % (int(pair[0]), int(pair[1]))):
            halo = save_halo_cutouts('TNG300-1', pair[0], pair[1])
            print('FP halo pulled')
            outname = 'cutout_fp_%d_%d.hdf5' % (pair[0], pair[1])
            with open(outname, 'wb') as f:
               f.write(halo.content)
            del(halo)
            gc.collect()

            halo = save_halo_cutouts('TNG300-1-Dark', pair[0], pair[2])
            outnamedm = 'cutout_dm_%d_%d.hdf5' % (pair[0], pair[2])
            with open(outnamedm, 'wb') as f:
               f.write(halo.content)
            del(halo)
            gc.collect()
            print('DM halo pulled')

            try:
               make_proj(outname, int(pair[0]), int(pair[1]), dm=False, gas=True)   
               os.remove(outname)

               # if not glob.glob('dm_massproj_%d_%d_z.fits' % (int(pair[0]), int(pair[2]))):
               make_proj(outnamedm, int(pair[0]), int(pair[1]), dm=True, gas=False)
               os.remove(outnamedm)
            except:
               print(pair, "error")
               continue

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