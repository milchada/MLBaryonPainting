import yt, soxs, pyxsim
import numpy as np 
import gc

def hot_gas(pfilter, data):
   pfilter1 = data[pfilter.filtered_type, "temperature"] > 3.0e5
   pfilter2 = data["PartType0", "StarFormationRate"] == 0.0
   pfilter3 = data[pfilter.filtered_type, "density"] < 5.0e-25
   return pfilter1 & pfilter2 & pfilter3

# We now tell yt what the particle filter is.
yt.add_particle_filter(
"hot_gas",
function=hot_gas,
filtered_type="gas",
requires=["temperature", "density"],
)

# This is a list of metal fields that we're going to allow to vary
# separately
var_elem = {
   elem: ("hot_gas", f"{elem}_fraction")
   for elem in ["He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]
}

emin = 0.5  # The minimum energy to generate in keV
emax = 2.0  # The maximum energy to generate in keV
nbins = 6000  # The number of energy bins between emin and emax
kT_max = 30.0  # The max gas temperature to use
source_model = pyxsim.CIESourceModel(
   "apec",
   emin,
   emax,
   nbins,
   ("hot_gas", "metallicity"),
   binscale="log",  # var_elem=var_elem,
   temperature_field=("hot_gas", "temperature"),
   emission_measure_field=("hot_gas", "emission_measure"),
   kT_max=kT_max,
)

exp_time = (100, "ks")  # exposure time
area = (2000.0, "cm**2")  # collecting area
sky_center = (45.0, 30.0)  # the center of the observation in RA, dec

def mocks(halonum, prefix='chandra_100ks', ns = ['x','y','z']):
   fn = 'cutout_%d.hdf5' % halonum
   ds = yt.load(fn)
   ds.add_particle_filter("hot_gas")

   _, c = ds.find_min(("nbody","Potential"))
   # width = ds.quan(5*r200[halonum], 'kpc')
   width = ds.quan(8, 'Mpc')
   le = c - 0.5 * width
   re = c + 0.5 * width
   box = ds.box(le, re)

   z = ds.current_redshift
   if z < 0.01:
      z = 0.01
   n_photons, n_cells = pyxsim.make_photons(
    'photons_%d' % halonum, box, z, area, exp_time, source_model
   )

   # this bit projects the photons along the z-axis and stores them in a file called "my_events.h5"
   # it also uses the wabs model for foreground galactic absorption with nH = 0.018
   for n in ns:
      n_events = pyxsim.project_photons(
          'photons_%d' % halonum,
          'events_%d_%s' % (halonum, n),
          n,
          sky_center,
          absorb_model="wabs",
          nH=0.018,
      )

      # This reads the events in my_events.h5 and creates a SIMPUT
      # file called cluster_simput.fits and a SIMPUT photon list
      # file called cluster_phlist.fits
      events = pyxsim.EventList('events_%d_%s.h5' % (halonum, n))
      events.write_to_simput('halo_%d_%s' % (halonum, n), overwrite=True)

      # This creates a mock Chandra observation of the cluster
      soxs.instrument_simulator(
          "halo_%d_%s_simput.fits" % (halonum,n),
          "%s_%d_%s.fits" % (prefix, halonum, n),
          exp_time,
          "chandra_acisi_cy0",
          sky_center,
          overwrite=True,
      )
      del(n_events, events)
      gc.collect(); gc.collect()

   del (ds, box)
   gc.collect(); gc.collect()
