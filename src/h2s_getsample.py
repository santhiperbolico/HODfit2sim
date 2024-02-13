"""
Generate samples of galaxies with a given number density

Parameters
-----------
filenom : string
    Name of the file with information on the "sample" simulation
ndtarget : list
    List of target number densities (nd)
propname  : list  
    List of property names to make cuts and obtain the target nd
verbose : boolean
    True to print the resolution
Testing : boolean
    True to reduce calculation
"""

import sys
import argparse
from src.h2s_cumuf import get_cumuf

# Defined command line options (clo) to handle lists
clo=argparse.ArgumentParser()

def list_of_strings(arg):
    return arg

clo.add_argument(
  "--listnd",  # Name on the command line
  nargs="*",   # 0 or more values expected => creates a list
  type=float,
  default=[-2.5],  # default if nothing is provided
)

clo.add_argument(
  "--listsim",
  nargs="*",
  type=list_of_strings, 
  default=['/users/arivgonz/output/Junk/BAHAMAS/HIRES/AGN_RECAL_nu0_L100N512_WMAP9/sample_snap31.hdf5'],
)

clo.add_argument(
  "--listprop",
  nargs="*",
  type=list_of_strings, 
  default=['Subhalo/Mass_030kpc'],
)

clo.add_argument(
  "--listbool",
  nargs="*",
  type=bool, 
  default=[True,True],
)

# Read the input arguments
args = clo.parse_args()

filenom = args.listsim[0]
nds = args.listnd
props = args.listprop
verbose, Testing = args.listbool
print(filenom); exit()
for prop in props:
    # Construct a cumulative function on the property
    cumuf = get_cumuf(prop,filenom,verbose=verbose,Testing=Testing)
    if verbose: print(f'Cumulative function for {prop}:\n {filenom}/{cumuf}')
#
## Get the edges of the halo mass bins
#edges = hmf.get_hmf(mhnom,npmin,dm,samplefile,verbose=verbose,Testing=Testing)
#if verbose: print("Number of haloes with masses from {:.2f} to {:.2f}".format(edges[0],edges[-1]))    

