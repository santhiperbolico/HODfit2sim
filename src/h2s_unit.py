import h2s_io as io
import h5py
import numpy as np

def get_galfiles(snap,sim,env):
    """
    Get the subfind files

    Parameters
    -----------
    snap : integer
        Snapshot number
    sims : list of strings
        Array with the names of the simulation
    env : string
        ari or cosma, to use the adecuate paths
 
    Returns
    -----
    files : array of string
       Subfind files with full paths
    allfiles : boolean
       True if all files encountered given the numbers in the files names

    Examples
    ---------
    >>> import h2s_bahamas as b
    >>> b.get_subfind_files(8,'L050N256/WMAP9/Sims/ex','cosma')
    >>> files, allfiles = b.get_subfind_files(27,'L400N1024/WMAP9/Sims/BAHAMAS','cosmalega')
    """

    allfiles = True
    if (env == 'laptop'):
        outff = ['/Users/Usuario/Documents/Master_Fisica_Teorica/TFM/UNITSIM1_model_z1.321_ELGs_l10000000.h5']

    return outff, allfiles


def get_subfind_prop(snap,sim,env,propdef,proptype=None,verbose=False,Testing=False,nfiles=2):
    """
    Get an array with a given property from the Subfind output

    Parameters
    -----------
    snap : integer
        Snapshot 
    sim : string
        Simulation name
    env : string
        ari or cosma, to use the adecuate paths
    propdef : string
        Name of the property, including path within hdf5 file
    proptype : string
        'DM', 'star', 'gas', 'BH', etc. for relevant properties
    Testing: boolean
        True or False
    nfiles : integer
        Number of files to be considered for testing
    verbose : boolean
        True to write first Subfind file out

    Returns
    -----
    prop : numpy array float
        Property within Subfind files

    Examples
    ---------
    >>> import h2s_bahamas as b
    >>> b.get_subfind_prop(27,'L400N1024/WMAP9/Sims/BAHAMAS','arilega',
                           'FOF/Group_M_Crit200',Testing=True)
    """

    # Simulation input
    files, allfiles = get_galfiles(snap,sim,env)
    if allfiles is False: return -999.
    if verbose: print('get_subfind_prop: First Subfind file is {}'.format(files[0]))

    if (proptype is not None):
        itype = ptypes.index(proptype)
        
    # Cycle through the files
    for ii,ff in enumerate(files):
        if (Testing and ii>=nfiles): break
        io.stop_if_no_file(ff)

        f = h5py.File(ff, 'r')
        if (ii == 0):
            if (proptype is None):
                try:
                    prop = f[propdef][:]
                except:
                    print('\n WARNING (bahamas): no {} found in {}'.format(propdef,ff))
                    return None
            else:
                try:
                    prop = f[propdef][:,itype]
                except:
                    print('\n WARNING (bahamas): no {} found in {}'.format(propdef,ff))
                    return None
        else:
            if (proptype is None):
                prop = np.append(prop, f[propdef][:], axis=0)
            else:
                prop = np.append(prop, f[propdef][:,itype], axis=0)
    return prop


def txt_to_h5(txt_file: str, h5_file: str, dataset_name: str = 'my_dataset', column_names: list[str] | None = None):
    """
    Convierte un archivo de texto (.txt) en un archivo HDF5 (.h5) usando h5py.
    
    Parámetros:
    -----------
    txt_file : str
        Ruta y nombre del archivo .txt de entrada (por ejemplo, 'datos.txt').
    h5_file : str
        Ruta y nombre del archivo .h5 de salida (por ejemplo, 'datos.h5').
    dataset_name : str, opcional
        Nombre del dataset dentro del archivo .h5. Por defecto es 'my_dataset'.
    
    Ejemplo de uso:
    ---------------
    txt_to_h5('datos.txt', 'datos.h5', dataset_name='mi_dataset')
    
    """
    # Lee los datos desde el fichero .txt
    data = np.loadtxt(txt_file)

    # Crea (o sobreescribe) el archivo HDF5 y el dataset dentro de él
    with h5py.File(h5_file, 'w') as h5f:
        dset = h5f.create_dataset(dataset_name, data=data)

        if column_names is not None:
            dset.attrs['column_names'] = column_names

    print(f"Archivo '{h5_file}' creado exitosamente con el dataset '{dataset_name}'.")

def suma(num1, num2):
    return num1 + num2