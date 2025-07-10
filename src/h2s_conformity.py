import numpy as np
from src.h2s_io import get_selection, create_dir
import h5py
import os

def load_halos(halo_file, file_format, col_pid, col_mass):
    """
    Loads the halo catalog and returns PID and mass arrays.

    Parameters
    ----------
    halo_file : str
        Path to the halo catalog file. Accepted formats: TXT (ASCII) or H5.
    file_format : {'txt', 'h5'}
        Format of the halo catalog. Use 'txt' for ASCII files or 'h5' for H5 datasets.
    col_pid : int or str
        If 'txt', the (0-based) column index for the PID field.
        If 'h5', the name of the dataset or field to use for PID.
    col_mass : int or str
        If 'txt', the (0-based) column index for the halo mass field.
        If 'h5', the name of the dataset or field to use for halo mass.

    Returns
    -------
    pid : numpy.ndarray
        Array of host PIDs, as loaded from the file.
    mass : numpy.ndarray
        Array of halo masses, as loaded from the file.

    Raises
    ------
    ValueError
        If the file format is not recognized or not supported.

    Notes
    -----
    For ASCII files, columns are assumed to be whitespace-separated and 0-based indexed.
    For H5 files, the function accesses datasets by their string names.
    """
    if file_format == "txt":
        halos = np.loadtxt(halo_file)
        pid   = halos[:, col_pid]
        mass  = halos[:, col_mass]
    elif file_format == "h5":
        with h5py.File(halo_file, "r") as f:
            pid  = f[col_pid][:]
            mass = f[col_mass][:]
    else:
        raise ValueError("halo_file format not recognised (use 'txt' or 'h5')")
    return pid, mass

def load_galaxies(galaxy_file, file_format, prop_host_id, prop_main_id, prop_mass):
    """
    Loads the galaxy catalog and returns HostHaloID, MainHaloID, and halo mass arrays.

    Parameters
    ----------
    galaxy_file : str
        Path to the galaxy catalog file. Accepted formats: TXT (ASCII) or H5.
    file_format : {'txt', 'h5'}
        Format of the galaxy catalog. Use 'txt' for ASCII files or 'h5' for H5 datasets.
    prop_host_id : int or str
        If 'txt', the (0-based) column index for HostHaloID.
        If 'h5', the name of the dataset or field to use for HostHaloID.
    prop_main_id : int or str
        If 'txt', the (0-based) column index for MainHaloID.
        If 'h5', the name of the dataset or field to use for MainHaloID.
    prop_mass : int or str
        If 'txt', the (0-based) column index for halo mass.
        If 'h5', the name of the dataset or field to use for halo mass.

    Returns
    -------
    host_id : numpy.ndarray
        Array of HostHaloID values, as loaded from the file.
    main_id : numpy.ndarray
        Array of MainHaloID values, as loaded from the file.
    mass : numpy.ndarray
        Array of halo masses, as loaded from the file.

    Raises
    ------
    ValueError
        If the file format is not recognized or not supported.

    Notes
    -----
    For ASCII files, columns are assumed to be whitespace-separated and 0-based indexed.
    For H5 files, the function accesses datasets by their string names.
    """
    if file_format == "txt":
        gal = np.loadtxt(galaxy_file)
        host_id = gal[:, prop_host_id]
        main_id = gal[:, prop_main_id]
        mass    = gal[:, prop_mass]
    elif file_format == "h5":
        with h5py.File(galaxy_file, "r") as f:
            host_id = f[prop_host_id][:]
            main_id = f[prop_main_id][:]
            mass    = f[prop_mass][:]
    else:
        raise ValueError("galaxy_file format not recognised (use 'txt' or 'h5')")
    return host_id, main_id, mass

def compute_conformity_parameters(
    halo_file,
    galaxy_file,
    M_min,
    M_max,
    N_bins,
    output_file,
    halo_format="txt",
    halo_pid = 13,       
    halo_mass = 17,       
    galaxy_format="h5",
    gal_host_id="HostHaloID", 
    gal_main_id="MainHaloID",
    gal_main_mass="MainMhalo",    
    verbose=True
):
    """
    Computes the satellite conformity parameters k1 and k2 as a function of halo mass.

    This function identifies central and satellite galaxies, separates satellites into
    those with/without a central galaxy in the same halo, bins all relevant quantities
    by halo mass, and computes conformity parameters per bin and globally. Results
    are saved in an output H5 file for further analysis.

    Parameters
    ----------
    halo_file : str
        Path to the input halo catalog (TXT or F5).
    galaxy_file : str
        Path to the input galaxy catalog (TXT or F5), previously filtered as needed.
    M_min : float
        Lower limit of halo mass (in log10 solar masses) for binning.
    M_max : float
        Upper limit of halo mass (in log10 solar masses) for binning.
    N_bins : int
        Number of mass bins to use for analysis.
    output_file : str
        Path for the H5 output file to store the results.
    halo_format : {'txt', 'h5'}, optional
        Format of the halo file. Default is 'txt'.
    halo_pid : int or str, optional
        Column index (if txt) or dataset name (if h5) for PID. Default is 13.
    halo_mass : int or str, optional
        Column index (if txt) or dataset name (if h5) for halo mass. Default is 17.
    galaxy_format : {'txt', 'h5'}, optional
        Format of the galaxy file. Default is 'h5'.
    gal_host_id : int or str, optional
        Column index (if txt) or dataset name (if h5) for HostHaloID.
    gal_main_id : int or str, optional
        Column index (if txt) or dataset name (if h5) for MainHaloID.
    gal_main_mass : int or str, optional
        Column index (if txt) or dataset name (if h5) for halo mass.
    verbose : bool, optional
        If True, prints progress and status messages.

    Returns
    -------
    None

    Outputs
    -------
    An H5 file is written to `output_file` containing:
        - Bin edges and counts for centrals, satellites, satellites with/without a central, and halos.
        - Per-bin k1 and k2 parameters.
        - Global k1 and k2 parameters.

    Raises
    ------
    ValueError
        If any file format or required field is not recognized.

    Notes
    -----
    Central galaxies are defined as those with HostHaloID == MainHaloID.
    Satellite galaxies are those with HostHaloID != MainHaloID.
    Satellites are further split into those with a central galaxy in their halo and those without.
    """

    # 1) Read & filter host halos (PID == -1 in col 13, logMvir from col 17)
    pid, mass = load_halos(halo_file, halo_format, halo_pid, halo_mass)
    mask_host = (pid == -1.0)
    logM_halos = np.log10(mass[mask_host])

    if verbose:
        print(f"Loaded {np.sum(mask_host)} host halos from {halo_file}")

    # 2) Read galaxies (no further flux cut!)
    host_id, main_id, mhalo = load_galaxies(galaxy_file, galaxy_format, gal_host_id, gal_main_id, gal_main_mass)
    mhalo = np.log10(mhalo)
    if verbose:
        print(f"Loaded {len(mhalo)} galaxies from {galaxy_file}")

    # 3) Classify centrals vs satellites
    is_c = (host_id == main_id)
    is_s = ~is_c

    C_ids = host_id[is_c]
    C_m   = mhalo[is_c]

    S_ids = main_id[is_s]
    S_m   = mhalo[is_s]

    sel_with    = np.isin(S_ids, C_ids)
    sel_without = ~sel_with

    S_mwc  = S_m[sel_with]
    S_mwoc = S_m[sel_without]

    # 4) Binning
    l_bin = (M_max - M_min) / N_bins

    M_min_bin, M_max_bin = [], []
    N_C, N_S, N_swc, N_swoc, N_H = [], [], [], [], []
    K1, K2 = [], []

    for i in range(N_bins):
        lo, hi = M_min + i*l_bin, M_min + (i+1)*l_bin

        n_c    = np.sum((C_m    >= lo)&(C_m    < hi))
        n_s    = np.sum((S_m    >= lo)&(S_m    < hi))
        n_swc  = np.sum((S_mwc  >= lo)&(S_mwc  < hi))
        n_swoc = np.sum((S_mwoc >= lo)&(S_mwoc < hi))
        n_h    = np.sum((logM_halos >= lo)&(logM_halos < hi))

        M_min_bin.append(lo)
        M_max_bin.append(hi)
        N_C.append(n_c)
        N_S.append(n_s)
        N_swc.append(n_swc)
        N_swoc.append(n_swoc)
        N_H.append(n_h)

        # per‐bin k1, k2
        if n_c>0 and n_s>0 and n_h>0:
            fracC = n_c / n_h
            k1b   = (n_swc * n_h) / (n_c * n_s)
            k2b   = (1 - k1b*fracC) / (1 - fracC) if (1 - fracC)!=0 else 0.0
        else:
            k1b = k2b = 0.0

        K1.append(k1b)
        K2.append(k2b)

    # 5) Global k1, k2
    N_C   = np.array(N_C)
    N_S   = np.array(N_S)
    N_swc = np.array(N_swc)
    N_swoc= np.array(N_swoc)
    N_H   = np.array(N_H)

    sum_c   = N_C.sum()
    sum_s   = N_S.sum()
    sum_swc = N_swc.sum()
    sum_h   = N_H.sum()

    Ncmv = np.array([ N_C[i]/N_H[i] if N_H[i]>0 else 0.0 for i in range(N_bins) ])

    if sum_s>0 and sum_h>0 and Ncmv.sum()>0:
        k1_global = sum(N_swc)/(sum(np.array(N_S) * np.array( Ncmv )))
        k2_global = sum(N_swoc)/(sum(np.array(N_S) * (1 - np.array(Ncmv))))
    else:
        k1_global = k2_global = 0.0

    if verbose:
        print(f"Global k1 = {k1_global:.4f}, k2 = {k2_global:.4f}")

    # 6) Save to H5
    create_dir(os.path.dirname(output_file))
    with h5py.File(output_file, "w") as out:
        grp = out.create_group("data")
        bins = grp.create_group("bins")
        bins.create_dataset("M_min_bin", data=np.array(M_min_bin))
        bins.create_dataset("M_max_bin", data=np.array(M_max_bin))
        bins.create_dataset("N_C",        data=N_C)
        bins.create_dataset("N_S",        data=N_S)
        bins.create_dataset("N_swc",      data=N_swc)
        bins.create_dataset("N_swoc",     data=N_swoc)
        bins.create_dataset("N_Halos",    data=N_H)
        bins.create_dataset("k1",         data=np.array(K1))
        bins.create_dataset("k2",         data=np.array(K2))

        glob = grp.create_group("global")
        glob.create_dataset("k1_global", data=k1_global)
        glob.create_dataset("k2_global", data=k2_global)

    if verbose:
        print("Results written to:", output_file)