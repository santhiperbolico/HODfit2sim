"""
Produce a file with those haloes to be considered, together with
the number of haloes per halo mass bin.

Parameters
-----------
simtype : string
    Simulation type (BAHAMAS, etc)
sim : string
    Simulation name
env : string
    ari, arilega or cosma, to use the adecuate paths
snap : float
    Simulation snapshot at the redshift of interest
mhname : string
    Name of the halo mass variable
npmin  : integer  
    Minimum number of particles within a halo, to be considered  
dm  : float
    Size of halo mass bin
dirz : string
    Path to table with z and snapshot.
dirout : string
    Path to output
verbose : boolean
    True to print the resolution
Testing : boolean
    True to reduce calculation

Examples
---------
>>> python3 h2s_gethaoes.py BAHAMAS HIRES/AGN_RECAL_nu0_L100N512_WMAP9 arilega 0.0 M200 20 0.1
"""

import sys
import h2s_io as io
import h2s_hmf as hmf

dirz = None
dirout = 'output/'
verbose = True
Testing = False

# Read the input arguments
simtype = sys.argv[1]
sim     = sys.argv[2]
env     = sys.argv[3]
snap    = int(sys.argv[4])
mhnom   = sys.argv[5]
npmin   = int(sys.argv[6])
dm      = float(sys.argv[7])
if len(sys.argv)>8:
    dirout  = sys.argv[8]
        
    if len(sys.argv)>9:
        verbose  = bool(sys.argv[9])
        
        if len(sys.argv)>10:
            Testing  = bool(sys.argv[10])

# Get name for the output file and populate it with a header
samplefile = io.generate_header(simtype,sim,env,snap,mhnom,dirout,filetype='sample')
if verbose: print(f'* Output file:\n   {samplefile}')

# Get the minimum halo mass to be considered
mhmin = hmf.get_mhmin(npmin,samplefile,verbose=verbose,Testing=Testing)
if verbose: print("* log10(Min. halo mass/Msun/h) = {:.2f}".format(mhmin))

# Write halo properties (ID, position and mass) within the new file
nhtot, nhdrop = hmf.write_halo_props(mhmin,samplefile,verbose=verbose,Testing=Testing)
if verbose: print(f'* Number of haloes dropped = {nhdrop}')    

# Write the halo mass function
edges = hmf.get_hmf(mhmin,dm,samplefile,verbose=verbose,Testing=Testing)
if verbose: print("* Number of haloes with masses from {:.2f} to {:.2f} = {}".format(edges[0],edges[-1],nhtot))    

