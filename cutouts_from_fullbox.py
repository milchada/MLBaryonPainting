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

def compile_training_data():
   import glob
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
