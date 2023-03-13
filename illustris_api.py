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

def save_halo_cutouts(simname, snapnum, nmin=0, nmax=100):    
    filenames = ['http://www.tng-project.org/api/%s/snapshots/%d/halos/%d/info.json' % (simname, snapnum, s) for s in range(nmin,nmax)]
    M200c = np.array([get(filename)['Group_M_Crit200']*munit.value for filename in filenames])
    print(len(M200c[M200c>1e14]))
    files = (np.array(filenames)[M200c > 1e14]).tolist()
    files = [f.replace('info.json', 'cutout.hdf5')for f in files] #.replace('-Dark','') --> this way its the FP file #
    dfiles = [f.replace('300-1', '300-1-Dark') for f in files]
    # print(files)

    for file in files+dfiles:
        if 'Dark' in file:
            prefix='dmo'
        else:
            prefix='fp'
        halonum = file.split('halos/')[1].split('/')[0]
        filename = 'cutout_%s_%d_%s.hdf5' % (prefix, snapnum, halonum)
        if not glob.glob(filename):    
            halo = get(file)
            with open(filename, 'wb') as f:
                    f.write(halo.content)
            print(halonum, 'done!')
            del(halo, f)
            gc.collect()

if __name__ == "__main__":
    for snapnum in snapnums:
        save_halo_cutouts(simname, snapnum, 0, 400)

def get_groupcat(simname, snapnum):
    filename = 'http://www.tng-project.org/api/%s/files/groupcat-%d/' % (simname, snapnum)

