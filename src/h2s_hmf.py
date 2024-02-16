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



def get_mhmin(npmin,filenom,verbose=True,Testing=False):
    """
    Get the miminum halo mass to be considered

    Parameters
    -----------
    npmin  : integer
        Minimum number of particles within a halo, to be considered
    filenom : string
        Name of the input and output file
    verbose : boolean
        True to print the resolution
    Testing: boolean
        True to reduce number of files and calculations

    Returns
    -----
    mp : float
        Minimum mass log10(Mass resolution in Msun/h) for DM haloes

    Examples
    ---------
    >>> import h2s_hmf as hmf
    >>> hmf.get_resolution('BAHAMAS','arilega','HIRES/AGN_TUNED_nu0_L050N256_WMAP9')
    """

    # Read properties from input file
    f = h5py.File(filenom, 'r')   
    header = f['header']
    simtype = header.attrs['simtype']
    sim     = header.attrs['sim']
    env     = header.attrs['workenv']
    snap    = header.attrs['snapshot']
    f.close()

    # Find minimum mass: mass resolution*minimum number of particles
    mp = 10.
    if not Testing:
        mp = get_mp(simtype,sim,env,snap,verbose=verbose)
    mmin = mp + np.log10(npmin)
            
    return mmin


def write_halo_props(mhmin,filenom,verbose=True,Testing=False):
    dsnom = 'haloes/'

    # Read properties from input file                                                             
    f = h5py.File(filenom, 'r')
    header = f['header']
    simtype = header.attrs['simtype']
    sim     = header.attrs['sim']
    env     = header.attrs['workenv']
    snap    = header.attrs['snapshot']
    mhnom   = header.attrs['mhnom']
    f.close()                  

    # Read halo properties
    match simtype:                                                                             
        case 'BAHAMAS':
            import src.h2s_bahamas as b
            mb = b.get_subfind_prop(snap,sim,env,mhnom,verbose=verbose,Testing=Testing)
            mh = b.mb2lmsun(mb,verbose=verbose) # log10(Msun/h)                 
            index = np.arange(0,len(mb),1)
            pos = b.get_subfind_prop(snap,sim,env,b.haloXYZnom,verbose=verbose,Testing=Testing)
            xh = pos[:,0]  # cMpc/h
            yh = pos[:,1]
            zh = pos[:,2]
            
        case other:
            print(f'Type of simulation not recognised: {simtype}'); return None

    # Initialise vectors
    mhalo, haloID, haloX, haloY, haloZ = [np.zeros(len(mh)) for i in range(5)]
    mhalo.fill(-999.); haloID.fill(-999.)
    
    # Impose minimum mass cut
    ind = np.where(mh > mhmin)
    nhdrop = len(mh) - np.shape(ind)[1]
    nhtot  = np.shape(ind)[1]
    if(np.shape(ind)[1] > 0):
        mhalo = mh[ind]; mh=[]
        haloID = index[ind]; index=[]
        haloX  = xh[ind];  xh=[]
        haloY  = yh[ind];  yh=[]
        haloZ  = zh[ind];  zh=[]        
        
    # Write output
    f = h5py.File(filenom, 'a')
    f.create_dataset(dsnom+'Mhalo', data=mhalo)
    f.create_dataset(dsnom+'haloID', data=haloID)
    f.create_dataset(dsnom+'haloX', data=haloX)
    f.create_dataset(dsnom+'haloY', data=haloY)
    f.create_dataset(dsnom+'haloZ', data=haloZ)    

    f[dsnom+'Mhalo'].dims[0].label = 'log10(M/Msun/h)'
    f[dsnom+'haloID'].dims[0].label = 'Index for main haloes'
    f[dsnom+'haloX'].dims[0].label = 'X coordinate for the halo center of potential (cMpc/h)'
    f[dsnom+'haloY'].dims[0].label = 'Y coordinate for the halo center of potential (cMpc/h)'
    f[dsnom+'haloZ'].dims[0].label = 'Z coordinate for the halo center of potential (cMpc/h)'    
    
    return nhtot,nhdrop



def get_hmf(mhmin,dm,filenom,verbose=True,Testing=False):    
    """
    Get the mass bins for a given simulatioin and redshift

    Parameters
    -----------
    mhmin  : float
        Minimum halo mass
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
    >>> hmf.get_hmf(12.,10.,'output_file.txt')
    """

    # Read properties from input file
    f = h5py.File(filenom, 'r')   
    mhalo  = f['haloes/Mhalo'][:]   # log10(M/Msun/h)
    f.close()

    # Find maximum mass
    mhmax = np.max(mhalo)

    # Define the limits of the mass bins
    medges = np.array(np.arange(mhmin,mhmax,dm))
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
        fileout = '/users/arivgonz/output/Junk/BAHAMAS/HIRES/AGN_RECAL_nu0_L100N512_WMAP9/example_mhGroupMCrit200_snap31.hdf5'
        
    #print(get_mp(simtype,sim,env,snap,dirz=dirz))
    #print(get_medges(20,simtype,sim,env,snap,'FOF/Group_M_Crit200',dirz=dirz,Testing=True))
    print(write_halo_props(10.,fileout,verbose=True,Testing=True))
