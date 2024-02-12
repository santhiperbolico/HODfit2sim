"""
Generate samples of galaxies with a given number density

Parameters
-----------
simtype : string
    Simulation type (BAHAMAS, etc)
sim : string
    Simulation name
env : string
    ari, arilega or cosma, to use the adecuate paths
zz : float
    Redshift to look
ndtarget : list
    List of target number densities (nd)
propname  : list  
    List of property names to make cuts and obtain the target nd
dirz : string
    Path to table with z and snapshot.
dirout : string
    Path to output
verbose : boolean
    True to print the resolution
Testing : boolean
    True to reduce calculation
"""

import sys
import argparse

# Defined command line options (clo) to handle lists
# this also generates --help and error handling
clo=argparse.ArgumentParser()
clo.add_argument(
  "--listnd",  # Name on the command line
  nargs="*",   # 0 or more values expected => creates a list
  type=float,
  default=[],  # default if nothing is provided
)

def list_of_strings(arg):
    return arg

clo.add_argument(
  "--listsim",
  nargs="*",
  type=list_of_strings, 
  default=[],
)

clo.add_argument(
  "--listprop",
  nargs="*",
  type=list_of_strings, 
  default=[],
)

clo.add_argument(
  "--listbool",
  nargs="*",
  type=bool, 
  default=[],
)


verbose = True
Testing = False

# Read the input arguments
args = clo.parse_args()

filenom = args.listsim
print(filenom,type(filenom)); exit()
nds = args.listnd
props = args.listprop
verbose, Testing = args.listbool



#for arg in args:
#    print(arg)
#simtype = sys.argv[1]
#sim     = sys.argv[2]
#env     = sys.argv[3]
#zz      = float(sys.argv[4])
#if len(sys.argv)>5:
#    dirz  = sys.argv[5]
#
#    if len(sys.argv)>6:
#        dirout  = sys.argv[6]
#        
#        if len(sys.argv)>7:
#            verbose  = bool(sys.argv[7])
#
#            if len(sys.argv)>8:
#                Testing  = bool(sys.argv[8])
#
#print(ndtarget)
#print(propname)
#print(dirz)
#          
## Get the snapshot corresponding to the input redshift 
#match simtype:                                                                         
#    case 'BAHAMAS':                                                                    
#        from src.h2s_bahamas import get_snap
#        snap, z_snap = get_snap(zz,sim,env,dirz=dirz)                                  
#                                                                                       
#    case other:                                                                        
#        print(f'Type of simulation not recognised: {simtype}'); exit()
#
## Get name for the output file and populate it with a header
#samplefile = io.generate_header(simtype,sim,env,zz,snap,dirout,filetype='sample')
#if verbose: print(f'{samplefile} is the output file')
#
## Get the edges of the halo mass bins
#edges = hmf.get_hmf(mhnom,npmin,dm,samplefile,verbose=verbose,Testing=Testing)
#if verbose: print("Number of haloes with masses from {:.2f} to {:.2f}".format(edges[0],edges[-1]))    

