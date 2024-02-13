import numpy as np
import h5py

def get_mp(simtype,sim,env,snap,verbose=True):
    """
    Get the mass resolution of a simulation

    Parameters
    -----------
    simtype : string
        Simulation type (BAHAMAS, etc)
    env : string
        ari, arilega or cosma, to use the adecuate paths
    sim : string
        Simulation name
    snap : integer
        Snapshot number corresponding to the input redshift
    verbose : boolean
        True to print the resolution

    Returns
    -----
    mp : float
        log10(Mass resolution in Msun/h) for DM and gas particles

    Examples
    ---------
    >>> import h2s_hmf as hmf
    >>> hmf.get_resolution('BAHAMAS','arilega','HIRES/AGN_TUNED_nu0_L050N256_WMAP9')
    """

    mp = 0.

    match simtype: 
        case 'BAHAMAS':
            from src.h2s_bahamas import get_resolution
            mp = get_resolution(sim,env,snap,verbose=verbose)

        case other:
            print(f'Type of simulation not recognised: {simtype}')
    return mp


def get_hmf(mhnom,npmin,dm,filenom,verbose=True,Testing=False):    
    """
    Get the mass bins for a given simulatioin and redshift

    Parameters
    -----------
    mhnom : string
        Name of the property, including path within hdf5 file
    npmin  : integer
        Minimum number of particles within a halo, to be considered
    dm  : float
        Size of the halo mass bin
    filenom : string
        Name of the input and output file
    verbose : boolean
        True to print the resolution
    Testing: boolean
        True to reduce number of files and calculations

    Returns
    -----
    medges : numpy array of floats
         Edges of the halo mass bins (Msun/h)

    Examples
    ---------
    >>> import h2s_hmf as hmf
    >>> hmf.get_medges('BAHAMAS','arilega',31,'FOF/Group_M_Crit200',20,'HIRES/AGN_TUNED_nu0_L050N256_WMAP9')
    """

    # Read properties from input file
    f = h5py.File(filenom, 'r')   
    header = f['header']
    simtype = header.attrs['simtype']
    sim     = header.attrs['sim']
    env     = header.attrs['workenv']
    snap    = header.attrs['snapshot']
    f.close()

    # Find minimum mass
    mp = 8.68
    if not Testing:
        mp = get_mp(simtype,sim,env,snap,verbose=verbose)
    mmin = mp + np.log10(npmin)

    # Find maximum mass such that there are at least 10 haloes within the given mass bin
    match simtype: 
        case 'BAHAMAS':
            from src.h2s_bahamas import get_subfind_prop, mb2lmsun
            mb = get_subfind_prop(snap,sim,env,mhnom,verbose=verbose,Testing=Testing)
            mhalo = mb2lmsun(mb,verbose=verbose)
            
        case other:
            print(f'Type of simulation not recognised: {simtype}'); return None

    mmax = np.max(mhalo)

    # Define the limits of the mass bins
    medges = np.array(np.arange(mmin,mmax,dm))
    elow  = medges[:-1]
    ehigh = medges[1:] 

    # Number of haloes
    nh  = np.zeros(shape=(len(elow)))  
    H, bins_edges = np.histogram(mhalo,bins=medges)
    nh[:] = nh[:] + H

    # Open output file to append dataset                                                      
    f = h5py.File(filenom, 'a')
    f.create_dataset('hmf/Mh_low',data=elow);
    f.create_dataset('hmf/Mh_high',data=ehigh);
    f.create_dataset('hmf/nh',data=nh);

    f['hmf/Mh_low'].dims[0].label = 'log10(M/Msun/h)'
    f['hmf/Mh_high'].dims[0].label = 'log10(M/Msun/h)'
    f['hmf/nh'].dims[0].label = 'Number of haloes per mass bin'     
    f.close()
    
    return medges


if __name__== "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath('..'))

    dirz = None ; outdir = None
    snap = 31 # zz = 0.

    simtype = 'BAHAMAS'
    env = 'arilega'
    if (env == 'arilega'):
        sim = 'HIRES/AGN_RECAL_nu0_L100N512_WMAP9'
        #sim = 'AGN_TUNED_nu0_L400N1024_WMAP9'
        dirz = '/users/arivgonz/output/BAHAMAS/'
        outdir = '/users/arivgonz/output/Junk/'

    #print(get_mp(simtype,sim,env,snap,dirz=dirz))
    print(get_medges(20,simtype,sim,env,snap,'FOF/Group_M_Crit200',dirz=dirz,Testing=True))

