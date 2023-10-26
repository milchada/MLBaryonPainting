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
