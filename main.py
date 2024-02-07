# Name of the simulation and work environment,
# Modify fithod_io.py  for your needs
sim = 'BAHAMAS' 
env = 'arilega' 

# Redshift, table with (z,snapnum), and output directory
z = []
dz_table =
output =

# Target number densities in log10(n/vol), as lists
ndtarget = [-2.5]

# Define the properties to impose cuts on, as lists
propname = ['']

# Define the halo mass bin size for <HODs>, shuffling and biasf 
dm0 = 0.057  #dex
dm_variable = True

get_sample = True
get_HOD = False

#--------------End of input parameters-------------------
import fit_sim4HOD.io as io
import preparenovonix.novonix_prep as prep


#--------------Start the calculations--------------------
if get_sample:
    # Halo mass range the mean HODs and shuffling 
    mmin = 50*mp
    #get_mranges(mmin,)

    # Construct sample of tracers into hdf5 files
    shuffle = True   # True for shuffling the sample

    
if get_HOD:
    #Read the info directly
    #if no adequate files message to run the code with get_sample = True
    print('get_HOD')
