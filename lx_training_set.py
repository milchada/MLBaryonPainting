import glob, os, gc
from astropy.io import fits
# from sklearn.preprocessing import QuantileTransformer

basePath = '/n/holylabs/LABS/natarajan_lab/Users/uchadaya/BaryonPasting/TNG-training-data/cutouts/'
dirs = ['highres/','midres/','lowres/','dmo/'] #'highres'

# def collect(dir):
	# os.chdir(basePath+dir)
lx = glob.glob('lx*fits'); lx.sort()
dm = glob.glob('dm*fits'); dm.sort()
rhog = glob.glob('gas_rho*fits'); rhog.sort()

toSort = True

if toSort:
	for file in lx+dm+rhog:
		hnum = file.split('halo')[1].split('_')[0]
		if int(hnum) < 10:
			fnew = file.replace('halo', 'halo00')
			os.rename(file, fnew)
		elif int(hnum) < 100:
			fnew = file.replace('halo', 'halo0')
			os.rename(file, fnew)

lx = glob.glob('lx*fits'); lx.sort()
dm = glob.glob('dm*fits'); dm.sort()
rhog = glob.glob('gas_rho*fits'); rhog.sort()

if len(lx):
	arrLx = np.zeros((len(lx), 512, 512))
	arrRhoG = np.zeros((len(lx), 512, 512))
	for i in range(len(lx)):
		arrLx[i] = fits.getdata(lx[i])
	np.save('lx_training.npy', arrLx)
	del(arrLx)
	gc.collect()

	for i in range(len(lx)):
		arrRhoG[i] = fits.getdata(rhog[i])
	np.save('rhog_training.npy', arrRhoG)
	del(arrRhoG)
	gc.collect()

arrDM = np.zeros((len(dm), 512, 512))
for i in range(len(dm)):
	arrDM[i] = fits.getdata(dm[i])

np.save('dm_training.npy', arrDM)
del(arrDM)
gc.collect()


for dir in dirs:
	collect(dir)

#then renorm and train as in autoencoder.py
#what about gas mass, temperature?
#like, try the gas mass function
