from astropy import units as u
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
import pyxsim
   
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

def yt_xray(snap, halo, subdir='', lx=True, dm=True, rho=True, temp=True):
    basePath = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/'+subdir
    ds = yt.load(basePath+'snap%d_halo_%d.hdf5' % (int(snap), int(halo)))#, default_species_fields="ionized")
    if lx:
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
            prj.writeto("dm_mass_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)
    if temp:
        for axis in ['x','y','z']:
            prj = yt.FITSProjection(ds, axis, ("gas","temperature"), width=(4.0, "Mpc"), center=c, weight_field="mazzotta_weighting")
            prj.writeto("gas_temp_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)
    if rho:
        for axis in ['x','y','z']:
            prj = yt.FITSProjection(ds, axis, ("PartType0","Density"), width=(4.0, "Mpc"), center=c)
            prj.writeto("gas_rho_snap%d_halo%d_%s.fits" % (snap, halo, axis), overwrite=True)

matches = np.load('../../lhalotree-matches.npy')
# cd cutouts/dmo
# cd cutouts/midres
# cd cutouts/lowres

# dirs = ['highres/']
# simnames = ['TNG300-1']

dir = 'highres/'
simname = 'TNG300-1'

error = []
def yt_continue():
    for (dir, simname) in zip (dirs, simnames):
        os.chdir('/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/'+dir)
        for snap in [50,67,78,99]:
            sub = matches[matches[:,0]==snap]
            hmax = int(sub[-1, 1])
            for halo in range(hmax):
                if not glob.glob('gas_temp_snap%d_halo%d_z.fits' % (snap, int(halo))):
                        print(dir, snap, halo)
                    # try:
                        save_halo_cutouts(simname, snap, halo, outname='snap%d_halo_%d.hdf5' % (snap, halo))
                        yt_xray(snap, halo,'cutouts/'+dir, lx=False, dm=False, rho=False, temp=True)
                        os.remove('snap%d_halo_%d.hdf5' % (snap, halo))
                        !rm *ewah
                # except:
                #     error.append('snap%d_halo_%d.hdf5' % (snap, halo))
                #     continue

def plot(fp, xmin=0, xmax=512, ncols=4):
    nrows = int(len(fp)/ncols)
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3, nrows*3))
    dx = fits.getheader(fp[0])['CDELT1']/1000. #Mpc per pix
    tix = np.arange(0, 4.1, .5)
    tpos = tix/dx
    tix -= 2
    zs = {'50':1, '67':0.5, '78': 0.3, '99': 0}
    i = 0
    for file in fp:
        f = fits.getdata(file)
        d = fits.getdata(file.replace('fp', 'dm'))
        snap = file.split('snap')[1].split('_')[0]
        halo = file.split('halo')[1].split('_')[0]
        axis = file.split('_')[-1].split('.')[0]
        z = zs[snap]
        ax.flatten()[i].imshow(f, cmap=cm.Greys, norm=colors.LogNorm())
        ax.flatten()[i+1].imshow(d, cmap=cm.Greys, norm=colors.LogNorm())
        ax.flatten()[i].text(xmin+50, xmin+50, r'z = %0.1f, halo ID = %s, $\hat{n}$ = %s' % (z, halo, axis))
        i += 2
    for a in ax.flatten():
        a.set_xticks(tpos)
        a.set_yticks(tpos)
    for a in ax[:,0]:
        a.set_yticklabels(['%0.1f' % x for x in tix])
    for a in ax[-1]:
        a.set_xticklabels(['%0.1f' % x for x in tix])
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.tight_layout()
    return fig, ax
 

def plot_npy(fp, ncols=4, rmax=1000, nbins=100):
    nrows = int(np.ceil(len(fp)*2/ncols))
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3, nrows*3))
    if 'Density' in fp[0]:
       ybins = 10**np.linspace(-7, 1, 81)
       ylabel = r'$\rho_{DM} (g/cm^3)$'
    if 'VelDisp' in fp[0]:
       ybins = np.linspace(0, 3000, 301)
       ylabel = r'$\sigma_v$ (km/s)'
    zs = {'50':1, '67':0.5, '78': 0.3, '99': 0}
    i = 0
    for file in fp:
        f = np.load(file)
        d = np.load(file.replace('fp', 'dm'))
        snap = file.split('snap')[1].split('_')[0]
        halo = file.split('halo')[1].split('_')[0]
        z = zs[snap]
        ax.flatten()[i].imshow(f.T, cmap=cm.viridis, norm=colors.LogNorm(f[f>0].min(), f.max()))
        ax.flatten()[i+1].imshow(d.T, cmap=cm.viridis, norm=colors.LogNorm(f[f>0].min(), f.max()))
        ax.flatten()[i].text(10,10, r'z = %0.1f, halo ID = %s' % (z, halo))
        i += 2
    x, y = d.shape
    plt.xlim(0, x)
    plt.ylim(0, y)
    x = np.arange(x)
    y = np.arange(y)
    r = np.linspace(0, 2000, 201)
    for a in ax.flatten():
        a.set_xticks(x[::20])
        a.set_yticks(y[::20])
    for a in ax[:,0]:
        a.set_yticklabels(['%0.0e' % x for x in ybins[:-1][::20]])
        a.set_ylabel(ylabel)
    for a in ax[-1]:
        a.set_xticklabels(['%d' % x for x in r[:-1][::20]])
        a.set_xlabel('R (kpc)')
    plt.tight_layout()
    return fig, ax
 
