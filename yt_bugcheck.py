import yt, glob

snapnum = 50
halonum = 10
simname = 'TNG300-1'
filename = 'snap%d_halo%d_fp.hdf5' % (snapnum, halonum)
halo = save_halo_cutouts(simname, snapnum, halonum, outname=filename)

ds = yt.load(filename)
_, c = ds.find_min(("PartType1", "Potential")) # center of the cylinder
L = [0, 0, 1] # vector along the length of the cylinder, here the z-axis
r = (1.0, "Mpc") # radius of cylinder
h = (20.0, "Mpc") # half-height of cylinder
dk = ds.disk(c, L, r, h)
print(dk["gas","mass"].sum())

prj = yt.FITSProjection(ds, 'z', ("gas", "density"), center=c, width=(2, "Mpc"))
rho = prj.get_data("density")
dx = (2000/512)*u.kpc.to('cm')
print((rho*(dx**2)).sum())