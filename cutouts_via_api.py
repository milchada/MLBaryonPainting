from astropy import units as u
import numpy as np
import requests, yt, gc, glob, os 

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"12345"}

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r

h = 0.6774 
munit = 1e10*u.Msun/h

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

from pathlib import Path
import h5py
import illustris_python as il
from yt import derived_field
import pandas as pd
import numpy as np
import yt, pyxsim
   
def hot_gas(pfilter, data):
    pfilter1 = data[pfilter.filtered_type, "temperature"] > 3.0e5
    pfilter2 = data["PartType0", "StarFormationRate"] == 0.0
    pfilter3 = data[pfilter.filtered_type, "density"] < 5e-25
    return pfilter1 & pfilter2 & pfilter3

# add the filter to yt itself
yt.add_particle_filter("hot_gas", function=hot_gas, filtered_type='gas', requires=["temperature","density"])
@derived_field(name="metals_solar", units="Zsun", sampling_type="local")
def _metals(field, data):
    return data["gas", "metallicity"] /0.0127

def yt_xray(snap, halo, subdir='', lx=True, dm=True, rho=True, temp=True, type='fp'):
    basePath = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/'+subdir
    ds = yt.load(basePath+'snap%d_halo%d_%s.hdf5' % (int(snap), int(halo), type))#, default_species_fields="ionized")
    if (lx or rho or temp):
        ds.add_particle_filter("hot_gas")
    # grp = groupcat[(groupcat['snap'] == snap)*(groupcat['halo'] == halo)].iloc[0]
    # c = [grp['GroupPosX'],grp['GroupPosY'],grp['GroupPosZ']]
    try:
        _, c = ds.find_min(("all", "Potential"))
    except:
        dd = ds.all_data() 
        try:
            pos = dd[('PartType1', 'Coordinates')]
            c = dd['PartType1','Coordinates'].mean(axis=0)
        except:
            pos = dd[('nbody', 'Coordinates')]
            c = dd['nbody','Coordinates'].mean(axis=0)
    if lx:
        source_model = pyxsim.CIESourceModel("apec", 0.1, 10.0, 1000,
                                        ("hot_gas","metallicity"),
                                        temperature_field=("hot_gas","temperature"),
                                        emission_measure_field=("hot_gas","emission_measure"))
        xray_fields = source_model.make_source_fields(ds, 0.5, 2.0)
        for axis in ['x','y','z']:
            prj = yt.FITSProjection(ds, axis, xray_fields[0], width=(4.0, "Mpc"), center=c)
            prj.writeto("lx_05_20_keV_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)
    if dm:
        for axis in ['x','y','z']:
            try:
                prj = yt.FITSParticleProjection(ds, axis, ("PartType1","Masses"), width=(4.0, "Mpc"), center=c, deposition="cic")
            except:
                prj = yt.FITSParticleProjection(ds, axis, ("nbody","Masses"), width=(4.0, "Mpc"), center=c, deposition="cic")
            if type=='fp':
                prj.writeto("dm_mass_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)
            else:
                prj.writeto("dm_mass_snap%d_halo%d_%s_dmo.fits" % (snap, halo, axis), overwrite=True)
    if temp:
        for axis in ['x','y','z']:
            prj = yt.FITSProjection(ds, axis, ("hot_gas","temperature"), width=(4.0, "Mpc"), center=c, weight_field=("hot_gas","mazzotta_weighting"))
            prj.writeto("gas_temp_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)
    if rho:
        for axis in ['x','y','z']:
            prj = yt.FITSProjection(ds, axis, ("hot_gas","density"), width=(4.0, "Mpc"), center=c)
            prj.writeto("gas_rho_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)

matches = np.load('fp_to_dm.npy')
import glob
simname = 'TNG300-1'
dir = 'highres/'

os.chdir('/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/'+dir)
for snap in [99]:#np.unique(matches[:,0])[:3]:
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
#            !rm *ewah
        #if not glob.glob('dm_mass_snap%d_halo%d_z_dm.fits' % (snap, halo)):
        #    if not glob.glob('snap%d_halo%d_dm.hdf5' % (snap, halo)):
        #        save_halo_cutouts(simname+'-Dark', snap, halo, outname='snap%d_halo%d_dm.hdf5' % (snap, halo))
        #    try:
        #        yt_xray(snap, halo,'cutouts/'+dir,lx=False,temp=False, dm=True,rho=False, type='dm')
        #    except:
        #        print('DM error')

