import numpy as np
import h5py
import sys
sys.path.append('/home2/guillermo/HODfit2sim')
import src.h2s_unit as hu
import src.h2s_io as io
import src.h2s_const as const

gal_file = '/home/aknebe/Projects/UNITSIM/SAMs/SAGE/ELGs/UNITSIM1/UNITSIM1_model_z1.321_ELGs.h5'
output_file = 'galaxy_cut.dat'

flux_cut = 1.041e-16

cutcols_gal = ['logFHalpha_att']
mincuts_gal = [np.log10(flux_cut)]
maxcuts_gal = [None]
selection_gal = io.get_selection(infile=gal_file, inputformat='hdf5',cutcols=cutcols_gal,mincuts=mincuts_gal,maxcuts=maxcuts_gal,testing=False,verbose=False)

params_gal = ['Xpos', 'Ypos', 'Zpos']
gal_data = io.read_data(infile=gal_file, cut=selection_gal, inputformat='hdf5', params=params_gal, testing=False, verbose=False)
Xposi = gal_data[0]
Yposi = gal_data[1]
Zposi = gal_data[2]
data = np.column_stack((Xposi, Yposi, Zposi))
np.savetxt(output_file, data, fmt='%.6e')
print(len(Xposi))