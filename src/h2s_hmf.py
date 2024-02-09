import numpy as np

def get_mp(simtype,sim,env,snap,dirz=None,verbose=True):
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
    dirz : string
        Path to table with z and snapshot.
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
            mp = get_resolution(sim,env,snap,dirz=dirz)

        case other:
            print(f'Type of simulation not recognised: {simtype}')
    return mp


def get_hmf(simtype,sim,env,snap,mhnom,npmin,dm,dirz=None,verbose=True,Testing=False):    
    """
    Get the mass bins for a given simulatioin and redshift

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
    mhnom : string
        Name of the property, including path within hdf5 file
    npmin  : integer
        Minimum number of particles within a halo, to be considered
    dirz : string
        Path to table with z and snapshot.
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

    # Find minimum mass
    mp = 1.
    if not Testing:
        mp = get_mp(simtype,sim,env,dirz=dirz,verbose=verbose)
    mmin = mp*npmin

    # Find maximum mass such that there are at least 10 haloes within the given mass bin
    match simtype: 
        case 'BAHAMAS':
            from src.h2s_bahamas import get_subfind_prop, mb2lmsun
            mb = get_subfind_prop(snap,sim,env,mhnom,Testing=Testing,nfiles=2,verbose=verbose)
            mhalo = mb2lmsun(mb,verbose=verbose)
            
        case other:
            print(f'Type of simulation not recognised: {simtype}'); return None

    nh = 0
    mmax = 20.

    # Define the limits of the mass bins
    medges = np.array(np.arange(mmin,mmax,dm))
    elow  = medges[:-1]
    ehigh = edges[1:] 

    # Number of haloes
    nh  = np.zeros(shape=(len(mhist)))  
    H, bins_edges = np.histogram(mhalo,bins=medges)
    nh[:] = nh[:] + H
    
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

