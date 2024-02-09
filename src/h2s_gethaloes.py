"""
Produce a file with those haloes to be considered, together with
the number of haloes per halo mass bin.

Parameters
-----------
npmin  : integer  
    Minimum number of particles within a halo, to be considered  
simtype : string
    Simulation type (BAHAMAS, etc)
env : string
    ari, arilega or cosma, to use the adecuate paths
sim : string
    Simulation name
zz : float
    Redshift to look
mhname : string
    Name of the halo mass variable
dirz : string
    Path to table with z and snapshot.
verbose : boolean
    True to print the resolution
Testing : boolean
    True to reduce calculation

Examples
---------
>>> python3 h2s_gethaoes.py 20 BAHAMAS HIRES/AGN_RECAL_nu0_L100N512_WMAP9 arilega 0.0 True
"""

import sys
import h2s_io as io
import h2s_hmf as hmf

dirz = None
verbose = True
Testing = False

# Read the input arguments
npmin   = int(sys.argv[1])
simtype = sys.argv[2]
sim     = sys.argv[3]
env     = sys.argv[4]
zz      = float(sys.argv[5])
mhnom   = sys.argv[6]
if len(sys.argv)>7:
    dirz  = sys.argv[7]

    if len(sys.argv)>8:
        verbose  = bool(sys.argv[8])

        if len(sys.argv)>9:
            Testing  = bool(sys.argv[9])
        
# Get name for the output file and populate it with a header

            
# Get the snapshot corresponding to the input redshift
match simtype:                                                                         
    case 'BAHAMAS':                                                                    
        from src.h2s_bahamas import get_snap
        snap, z_snap = get_snap(zz,sim,env,dirz=dirz)                                  
                                                                                       
    case other:                                                                        
        print(f'Type of simulation not recognised: {simtype}'); exit()

# Get the edges of the halo mass bins
edges = hmf.get_medges(npmin,simtype,sim,env,snap,mhnom,dirz=dirz,verbose=verbose,Testing=Testing)
print(edges)    

