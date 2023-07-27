from astropy import units as u
import numpy as np
import requests, yt, gc, glob, os 

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"5e8664e232c2b2ba3fe7a37b7977f31e"}

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

