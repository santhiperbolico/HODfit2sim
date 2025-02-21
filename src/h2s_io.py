import os.path
import sys
import h5py
import numpy as np
import h2s_const as const

#class nf(float):
#    '''
#    Define a class that forces representation of float to look a certain way
#    This removes trailing zero so '1.0' becomes '1'
#    '''
#    def __repr__(self):
#        str = '%.1f' % (self.__float__(),)
#        if str[-1] == '0':
#            return '%.0f' % self.__float__() 
#        else:
#            return '%.1f' % self.__float__()
    

def stop_if_no_file(infile):
    '''
    Stop if the file does not exist
    '''
    if (not os.path.isfile(infile)):
        print('STOP: no input file {}'.format(infile)) 
        sys.exit()
    return


def check_file(infile,verbose=False):
    '''
    Return True if the file exists
    '''
    file_fine = True  
    if (not os.path.isfile(infile)):
        file_fine = False
        if verbose:
            print('WARNING (io.check_file): file not found {}'.format(infile))

    return file_fine


def create_dir(dirout):
    '''
    Return True if directory already exists or it has been created
    '''
    if not os.path.exists(dirout):
        try:
            os.makedirs(dirout)
        except:
            print('WARNING (h2s_io.create_dir): problem creating directory ',dirout)
            return False
    return True


#def is_sorted(a):
#    '''
#    Return True if the array is sorted
#    '''
#    for i in range(len(a)-1): 
#        if a[i+1] < a[i] : 
#            return False
#    return True


def print_h5attr(infile,inhead='Header'):
    """
    Print out the group attributes of a hdf5 file

    Parameters
    ----------
    infile : string
      Name of input file (this should be a hdf5 file)
    inhead : string
      Name of the group to read the attributes from

    Example
    -------
    >>> import h2s_io as io
    >>> infile = '/hpcdata0/simulations/BAHAMAS/AGN_TUNED_nu0_L100N256_WMAP9/Data/Snapshots/snapshot_026/snap_026.27.hdf5'
    >>> io.print_h5attr(infile,inhead='Units')
    """

    filefine = check_file(infile) #print(filefine)
    if (not filefine):
        print('WARNING (h2s_io.printh5attr): Check that the file provided is correct')
        return ' '
    
    f = h5py.File(infile, 'r')
    header = f[inhead]
    for hitem in list(header.attrs.items()): 
        print(hitem)
    f.close()

    return ' '


#def count_symbol(infile,sym):
#    fileok = check_file(infile)
#    if fileok:
#        total = 0
#        with  open(infile, "r") as f:
#            count = sum(line.count(sym) for line in f)
#            total += count
#        print('There are {} {} in {}'.format(count,sym,infile))
#    else:
#        print('File {} not found'.format(infile))
#    return ' '


def get_file_name(simtype,sim,snap,mhnom,dirout,filetype='example'):
    """
    Get the name of the file containing the samples information

    Parameters
    -----------
    simtype : string
        String with the type of simulation (e.g. BAHAMAS)
    sim : string
        Name of the simulation
    snap : integer
        Snapshot number for the redshift of interest
    mhnom : string 
        Name of the halo mass to be used
    dirout : string
        Path to output
    filetype : string
        sample, HODgal, ...
 
    Returns
    -----
    filenom : string
       Full path to the sample file

    Examples
    ---------
    >> from h2s_io import get_file_name
    >> get_file_name('BAHAMAS','L050N256/WMAP9',33,'FOF/Group_M_Crit200','/users/arivgonz/output/Junk/')
    """
    
    path2file = dirout+simtype+'/'+sim
    # Generate the directory if needed
    create_dir(path2file)

    # Include the name of the halo mass in the path
    mpath = mhnom.replace("_","")
    if ('/' in mhnom):
        mpath = mpath.split('/')[1]

    filenom = path2file+'/'+filetype+'_mh'+mpath+'_snap'+str(snap)+'.hdf5'
    
    return filenom
    


def generate_header(simtype,sim,env,snap,mhnom,dirout,filetype='example'):
    """
    Get the name of the file containing the samples information

    Parameters
    -----------
    simtype : string
        String with the type of simulation (e.g. BAHAMAS)
    sim : string
        Name of the simulation
    env : string
        Working environment
    snap : integer
        Snapshot number for the redshift of interest
    mhnom : string 
        Name of the halo mass to be used
    dirout : string
        Path to output
    filetype : string
        sample, HODgal, ...
 
    Returns
    -----
    filenom : string
       Full path to the sample file

    Examples
    ---------
    >>> import h2s_io as io
    >>> io.generate_header('BAHAMAS','L050N256/WMAP9','arilega',0,31,'/users/arivgonz/output/Junk/')
    """

    # Get the file name
    filenom = get_file_name(simtype,sim,snap,mhnom,dirout,filetype=filetype)

    # Get cosmology, the simulation box side in Mpc/h and the redshift
    match simtype:                                                                             
        case 'BAHAMAS':                                                                        
            from h2s_bahamas import get_cosmology, get_z
            omega0, omegab, lambda0, h0, boxside = get_cosmology(sim,env)
            zz = get_z(snap,sim,env)
            
        case other:                                                                            
            print(f'Type of simulation not recognised: {simtype}')
            return None
    
    # Generate the output file (the file is rewrtitten)
    hf = h5py.File(filenom, 'w')

    # Generate a header
    headnom = 'header'
    head = hf.create_dataset(headnom,(100,))
    head.attrs[u'simtype']      = simtype
    head.attrs[u'sim']          = sim
    head.attrs[u'workenv']      = env
    head.attrs[u'snapshot']     = snap
    head.attrs[u'redshift']     = zz
    head.attrs[u'omega0']       = omega0
    head.attrs[u'omegab']       = omegab
    head.attrs[u'lambda0']      = lambda0        
    head.attrs[u'h0']           = h0
    head.attrs[u'boxside']      = boxside  #Mpc/h
    head.attrs[u'mhnom']        = mhnom
    hf.close()
    
    return filenom
    
def get_nheader(infile,firstchar=None):
    '''
    Given a text file with a structure: header+data, 
    counts the number of header lines

    Parameters
    -------
    infile : string
        Input file

    Returns
    -------
    ih : integer
        Number of lines with the header text
    '''


    ih = 0
    with open(infile,'r') as ff:
        for line in ff:
            if not line.strip():
                # Count any empty lines in the header
                ih += 1
            else:
                sline = line.strip()
                
                # Check that the first character is not a digit
                char1 = sline[0]
                word1 = sline.split()[0]
                if not firstchar:
                    if (not char1.isdigit()):
                        if (char1 != '-'):
                            ih += 1
                        else:
                            try:
                                float(word1)
                                return ih
                            except:
                                ih += 1
                    else:
                        return ih
                else:
                    if char1 == firstchar:
                        ih+=1
    return ih
        


def get_selection(infile, inputformat='hdf5',
                  cutcols=None, mincuts=[None], maxcuts=[None],
                  testing=False,verbose=False):
    '''
    Get indexes of selected galaxies

    Parameters
    ----------
    infile : strings
     List with the name of the input files. 
     - In text files (*.dat, *txt, *.cat), columns separated by ' '.
     - In csv files (*.csv), columns separated by ','.
    inputformat : string
     Format of the input file.
    cutcols : list
     Parameters to look for cutting the data.
     - For text or csv files: list of integers with column position.
     - For hdf5 files: list of data names.
    mincuts : strings
     Minimum value of the parameter of cutcols in the same index. All the galaxies below won't be considered.
    maxcuts : strings
     Maximum value of the parameter of cutcols in the same index. All the galaxies above won't be considered.
    verbose : boolean
      If True print out messages
    testing : boolean
      If True only run over few entries for testing purposes

    Returns
    -------
    selection : array of integers
    '''

    selection = None
    
    check_file(infile, verbose=verbose)

    if testing:
        #limit = const.testlimit
        limit = 50
    else:
        limit = None    

    if inputformat not in const.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(const.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            ind = np.arange(len(hf[cutcols[0]][:]))
            for i in range(len(cutcols)):
                if cutcols[i]:
                    param = hf[cutcols[i]][:]
                    mincut = mincuts[i]
                    maxcut = maxcuts[i]

                    if mincut and maxcut:
                        ind = np.intersect1d(ind,np.where((mincut<param)&(param<maxcut))[0])
                    elif mincut:
                        ind = np.intersect1d(ind,np.where(mincut<param)[0])
                    elif maxcut:
                        ind = np.intersect1d(ind,np.where(param<maxcut)[0])
            selection = ind[:limit]
    elif inputformat=='txt':
        ih = get_nheader(infile)
        ind = np.arange(len(np.loadtxt(infile,usecols=cutcols[0],skiprows=ih)))
        for i in range(len(cutcols)):
            if cutcols[i]:
                param = np.loadtxt(infile,usecols=cutcols[i],skiprows=ih)
                mincut = mincuts[i]
                maxcut = maxcuts[i]

                if mincut and maxcut:
                    ind = np.intersect1d(ind,np.where((mincut<param)&(param<maxcut))[0])
                elif mincut:
                    ind = np.intersect1d(ind,np.where(mincut<param)[0])
                elif maxcut:
                    ind = np.intersect1d(ind,np.where(param<maxcut)[0])
        selection = ind[:limit]
    else:
        if verbose:
            print('STOP (gne_io.get_selection): ',
                  'Input file has not been found.')
        sys.exit()

    return selection

def read_data(infile, cut, inputformat='hdf5', params=[None],
              testing=False, verbose=True):    
    '''
    Read input data per column/dataset
    
    Parameters
    ----------
    infile : string
       Name of the input file. 
    cut : array of integers
       List of indexes of the selected galaxies from the samples.
    inputformat : string
       Format of the input file.
    params : list of either integers or strings
       Inputs columns for text files or dataset name for hdf5 files.
    testing : boolean
       If True only run over few entries for testing purposes
    verbose : boolean
       If True print out messages.
     
    Returns
    -------
    outparams : array of floats
    '''

    check_file(infile, verbose=verbose)

    if inputformat not in const.inputformats:
        if verbose:
            print('STOP (gne_io): Unrecognised input format.',
                  'Possible input formats = {}'.format(const.inputformats))
        sys.exit()
    elif inputformat=='hdf5':
        with h5py.File(infile, 'r') as hf:
            ii = 0
            for nomparam in params:
                if (nomparam is not None):
                    try:
                        ###here what if Pos/vel as matrix?
                        prop = hf[nomparam][cut]
                    except:
                        print('\n WARNING (gne_io): no {} found in {}'.format(
                            nomparam,infile))

                    if (ii == 0):
                        outparams = prop
                    else:
                        outparams = np.vstack((outparams,prop))
                    ii += 1

    elif inputformat=='txt': ###need to adapt to the generalisation and test
        ih = get_nheader(infile)
        outparams = np.loadtxt(infile,skiprows=ih,usecols=params)[cut].T

    return outparams

if __name__== "__main__":
   
    infile = 'blu'
    dirout = 'remove_dir/blu'

    #print(nf('4.0'))
    #print('Check file {}: {}'.format(infile,check_file(infile)))
    #print('Create dir {}: {}'.format(dirout,create_dir(dirout)))
    #print(is_sorted([1,3,6]))
    #print(is_sorted([1,9,6]))
    #print(count_symbol('/home/violeta/Downloads/blu',';'))
    #print(stop_if_no_file(infile))

    #-------------Test generate_header---------------------------
    import os, sys                                                                         
    sys.path.insert(0, os.path.abspath('..'))
    simtype = 'BAHAMAS'
    sim = 'HIRES/AGN_RECAL_nu0_L100N512_WMAP9'; env = 'arilega'
    snap = 31
    mhnom = 'FOF/Group_M_Crit200'
    dirout = '/users/arivgonz/output/Junk/'
    print(generate_header(simtype,sim,env,snap,mhnom,dirout))

