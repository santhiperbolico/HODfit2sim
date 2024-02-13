import numpy as np
import h5py

def get_cumuf(prop,filenom,proptype=None,verbose=True,Testing=False):    
    """
    Get the cumulative function for a given property

    Parameters
    -----------
    prop : string
        Name of the property, including path within hdf5 file
    proptype : string
        'DM', 'star', 'gas', 'BH', etc. for relevant properties 
    filenom : string
        Name of the input and output file
    verbose : boolean
        True to print the resolution
    Testing: boolean
        True to reduce number of files and calculations

    Returns
    -----
    cfnom : string
         Name of the data set containing the cumulative function

    Examples
    ---------
    >>> from h2s_cumuf import get_cumuf 
    >>> get_cumuf('Subhalo/Mass_030kpc','sample_snap31.hdf5')
    """

    # Read properties from input file
    f = h5py.File(filenom, 'r')   
    header = f['header']
    simtype = header.attrs['simtype']
    sim     = header.attrs['sim']
    env     = header.attrs['workenv']
    snap    = header.attrs['snapshot']
    mhmin   = f['hmf/Mh_low'][0]
    mhmax   = f['hmf/Mh_high'][-1]
    f.close()

    # Read the property
    match simtype: 
        case 'BAHAMAS':
            from src.h2s_bahamas import get_subfind_prop, mb2lmsun
            val1 = get_subfind_prop(snap,sim,env,prop,proptype=proptype,
                                      Testing=Testing,verbose=verbose)
            if ('mass' in prop or 'Mass' in prop):
                print(val1[0:5])
                vals = mb2lmsun(val1,verbose=verbose)
            else:
                vals = val1
            
        case other:
            print(f'Type of simulation not recognised: {simtype}'); return None

    pmin = max = np.max(mhalo)
    #
    ## Define the limits of the mass bins
    #medges = np.array(np.arange(mmin,mmax,dm))
    #elow  = medges[:-1]
    #ehigh = medges[1:] 
    #
    ## Number of haloes
    #nh  = np.zeros(shape=(len(elow)))  
    #H, bins_edges = np.histogram(mhalo,bins=medges)
    #nh[:] = nh[:] + H
    #
    ## Open output file to append dataset                                                      
    #f = h5py.File(filenom, 'a')
    cfnom = '..'    
    #f.create_dataset('hmf/Mh_low',data=elow);
    #f.create_dataset('hmf/Mh_high',data=ehigh);
    #f.create_dataset('hmf/nh',data=nh);
    #
    #f['hmf/Mh_low'].dims[0].label = 'log10(M/Msun/h)'
    #f['hmf/Mh_high'].dims[0].label = 'log10(M/Msun/h)'
    #f['hmf/nh'].dims[0].label = 'Number of haloes per mass bin'     
    #f.close()

    return cfnom


if __name__== "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath('..'))

    prop = 'Subhalo/Mass_030kpc'
    filenom = '/users/arivgonz/output/Junk/BAHAMAS/HIRES/AGN_RECAL_nu0_L100N512_WMAP9/sample_snap31.hdf5'
    print(get_cumuf(prop,filenom,proptype='star'))


