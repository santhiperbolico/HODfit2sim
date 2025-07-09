import numpy as np
import h5py
import sys
sys.path.append('/home2/guillermo/HODfit2sim')
import src.h2s_unit as hu
import src.h2s_io as io
import src.h2s_const as const
import sys

def get_params_gal(halo_file, snap, sim, env, flux_cut, M_min, M_max, N_bin, output_file, testing = False, verbose = True):

    """
    Read halos (text files) and galaxies (HDF5 files),
    applies flux selection, separates into centrals/satellites,
    groups into mass bins and calculates k1, k2 conformity parameters.
    Then, writes the results in an HDF5 file.

    Parameters
    ----------
    halo_file : str
        Path to the .txt file containing halo information.
    snap : int
        Snapshot number to read (for galaxy file).
    sim : str
        Simulation name (for galaxy file).
    env : str
        Environment ('laptop', etc) .
    flux_cut : float
        Flux threshold (log10(flux_cut) will be used to filter logFHalpha_att).
    M_min : float
        Minimum of the logarithmic mass scale for binning.
    M_max : float
        Maximum of the logarithmic mass scale for binning.
    N_bin : int
        Number of bins.
    output_file : str
        Path to the output HDF5 file where results will be stored.
    testing : boolean, optional
        True or False.
    verbose : boolean, optional
        If True, prints trace messages (default is True).

    Returns
    ----------
    None
        The results are written to the specified HDF5 file.
    """

    # ------------------------------------------------------------------------
    # 1) HALOS READING (text file)
    # ------------------------------------------------------------------------
    if verbose:
        print(f"[get_params_gal] Reading halos from: {halo_file}")
    # Trivial selection to read all rows
    cutcols_halo = [0]     # One column enough to count rows
    mincuts_halo = [None]
    maxcuts_halo = [None]
    selection_halo = io.get_selection(infile=halo_file,
                                      inputformat='txt',
                                      cutcols=cutcols_halo,
                                      mincuts=mincuts_halo,
                                      maxcuts=maxcuts_halo,
                                      testing=testing,
                                      verbose=verbose)

    # Read relevant columns for halos (adjust indexes as needed)
    cols_halo_to_read = [1, 4, 5, 6, 7, 8, 12]
    halos_data = io.read_data(infile=halo_file,
                              cut=selection_halo,
                              inputformat='txt',
                              params=cols_halo_to_read,
                              testing=testing,
                              verbose=verbose)
    # halos_data -> shape (n_cols, n_halos)
    ID_Halo = halos_data[0]
    Mvir = halos_data[-1]  # Last column, sup. col 12 = Mass
    logMvir = np.log10(Mvir)


    # ------------------------------------------------------------------------
    # 2) GALAXY READING (HDF5 file)
    # ------------------------------------------------------------------------
    gal_files, allfiles = hu.get_galfiles(snap, sim, env)
    if (not allfiles) or (len(gal_files) == 0):
        print(f"[get_params_gal] No galaxy files found"
              f"for snap={snap}, sim='{sim}', env='{env}'.")
        sys.exit(1)

    # Assuming there is only one file in 'laptop'
    gal_file = gal_files[0]
    if verbose:
        print(f"[get_params_gal] Reading galaxies from: {gal_file}")


    # ------------------------------------------------------------------------
    # 3) FLUX CUT FOR GALAXIES
    # ------------------------------------------------------------------------

    cutcols_gal = ['logFHalpha_att']
    mincuts_gal = [np.log10(flux_cut)]
    maxcuts_gal = [None]
    selection_gal = io.get_selection(infile=gal_file,
                                     inputformat='hdf5',
                                     cutcols=cutcols_gal,
                                     mincuts=mincuts_gal,
                                     maxcuts=maxcuts_gal,
                                     testing=testing,
                                     verbose=verbose)

    # We read HostHaloID, MainHaloID, MainMhalo
    params_gal = ['HostHaloID', 'MainHaloID', 'MainMhalo']
    gal_data = io.read_data(infile=gal_file,
                            cut=selection_gal,
                            inputformat='hdf5',
                            params=params_gal,
                            testing=testing,
                            verbose=verbose)
    # gal_data -> shape (3, n_selected_gal)
    data0F = gal_data[0]  # HostHaloID
    data1F = gal_data[1]  # MainHaloID
    data2F = np.log10(gal_data[2])  # log10(MainMhalo)

    # ------------------------------------------------------------------------
    # 4) CLASSIFICATION: CENTRALS / SATELLITES
    # ------------------------------------------------------------------------
    is_sat = (data0F != data1F)
    is_cen = (data0F == data1F)

    SM = data1F[is_sat]     # Halo Main ID of satellites
    SM_M = data2F[is_sat]   # mass satellites
    C = data0F[is_cen]      # Halo host ID of centrals
    C_M = data2F[is_cen]    # masa central


    # ------------------------------------------------------------------------
    # 5) SATELLITES WITH / WITHOUT CENTRAL
    # ------------------------------------------------------------------------
    selwc = np.isin(SM, C)     # satellites with central
    selwoc = ~selwc
    SM_Mwc  = SM_M[selwc]
    SM_Mwoc = SM_M[selwoc]


    # ------------------------------------------------------------------------
    # 6) BINNING IN HALO MASS
    # ------------------------------------------------------------------------
    l_bin = (M_max - M_min) / N_bin

    M_min_bin  = []
    M_max_bin  = []
    N_C        = []
    N_S        = []
    N_swc      = []
    N_swoc     = []
    N_Halos    = []
    K1         = []
    K2         = []

    ii = M_min
    while ii < M_max:

        minf = ii
        msup = ii + l_bin

        # Centrals in bin
        mask_c = (C_M >= minf) & (C_M < msup)
        N_C_bin = mask_c.sum()

        # Satellites in bin
        mask_s = (SM_M >= minf) & (SM_M < msup)
        N_S_bin = mask_s.sum()

        # Sat. with central in bin
        mask_swc = (SM_Mwc >= minf) & (SM_Mwc < msup)
        N_Swc_bin = mask_swc.sum()

        # Sat. without central in bin
        mask_swoc = (SM_Mwoc >= minf) & (SM_Mwoc < msup)
        N_Swoc_bin = mask_swoc.sum()

        # Halos in bin
        mask_halo = ((logMvir >= minf) & (logMvir < msup))
        N_halos_bin = mask_halo.sum()

        if (N_C_bin > 0 or N_S_bin > 0):
            M_min_bin.append(minf)
            M_max_bin.append(msup)
            N_C.append(N_C_bin)
            N_S.append(N_S_bin)
            N_swc.append(N_Swc_bin)
            N_swoc.append(N_Swoc_bin)
            N_Halos.append(N_halos_bin)
            K1.append(k1_bin)
            K2.append(k2_bin)

        # k1, k2 in this bin
        if N_halos_bin > 0:
            if (N_C_bin>0 and N_S_bin>0):
                k1_bin = (N_Swc_bin) * (1./N_C_bin) * (1./N_S_bin) * N_halos_bin
                fracC  = N_C_bin / float(N_halos_bin)

                if (1 - fracC) != 0:
                    k2_bin = (1 - k1_bin*fracC) / (1 - fracC)
                else:
                    k2_bin = 0.0
            else:
                k1_bin = 0.0
                k2_bin = 0.0
        else:
            k1_bin = 0.0
            k2_bin = 0.0

        ii += l_bin


    # ------------------------------------------------------------------------
    # 7) k1_global AND k2_global
    # ------------------------------------------------------------------------
    sum_swc = np.sum(N_swc)
    sum_swoc = np.sum(N_swoc)
    sum_c = np.sum(N_C)
    sum_s = np.sum(N_S)
    sum_h = np.sum(N_Halos)

    # Original formula:
    # k1_global = sum(N_swc)*sum(N_Halos)/( sum(N_S)* sum(N_cm_v) )
    # N_cm_v = N_C_bin / N_halos_bin in each bin. 
    N_cm_v = []
    for ic in range(len(N_C)):
        if N_Halos[ic] > 0:
            N_cm_v.append(N_C[ic] / float(N_Halos[ic]))
        else:
            N_cm_v.append(0.0)
    N_cm_v = np.array(N_cm_v)

    if sum_s > 0 and sum_h>0 and N_cm_v.sum()>0:
        k1_global = (sum_swc * sum_h) / ( sum_s * N_cm_v.sum() )
        k1_global_1 = sum_swc/(np.array(N_S) * N_cm_v).sum()
        fracC_all = sum_c / float(sum_h)
        if (1 - fracC_all) != 0:
            k2_global = (1 - k1_global*fracC_all) / (1 - fracC_all)
            k2_global_1 = sum_swoc/(np.array(N_S) * (1-N_cm_v)).sum()
        else:
            k2_global = 0.0
            k2_global_1 = 0.0
    else:
        k1_global = 0.0
        k2_global = 0.0
        k1_global_1 = 0.0
        k2_global_1 = 0.0

    if verbose:
        print(f"[get_params_gal] Total: N_C={sum_c}, N_S={sum_s}, "
              f"N_swc={sum_swc}, N_swoc={sum_swoc}, N_Halos={sum_h}")
        print(f"[get_params_gal] k1_global={k1_global_1}, "
              f"k2_global={k2_global_1}")
        
    # ------------------------------------------------------------------------
    # 8) WRITE RESULTS TO HDF5 FILE
    # ------------------------------------------------------------------------
    with h5py.File(output_file, 'w') as f:
        # Create main group named 'data'
        grp = f.create_group('data')

        # Create sub-group 'bins' for binned data
        bins_grp = grp.create_group('bins')
        bins_grp.create_dataset('M_min_bin', data=np.array(M_min_bin))
        bins_grp.create_dataset('M_max_bin', data=np.array(M_max_bin))
        bins_grp.create_dataset('N_C', data=np.array(N_C))
        bins_grp.create_dataset('N_S', data=np.array(N_S))
        bins_grp.create_dataset('N_swc', data=np.array(N_swc))
        bins_grp.create_dataset('N_swoc', data=np.array(N_swoc))
        bins_grp.create_dataset('N_Halos', data=np.array(N_Halos))
        bins_grp.create_dataset('k1', data=np.array(K1))
        bins_grp.create_dataset('k2', data=np.array(K2))

        # Create sub-group 'global' for global data
        global_grp = grp.create_group('global')
        global_grp.create_dataset('k1_global', data=k1_global_1)
        global_grp.create_dataset('k2_global', data=k2_global_1)

    if verbose:
        print(f"[get_params_gal] Results written to: {output_file}")

    return output_file

#/home2/guillermo/out_90p_UNIT1_fnl100_1Gpc_x_y_z_vx_vy_vz_M_.txt   Condiciones iniciales distintas
#/home2/guillermo/Halos_tree_DOC_PID_Vmax_all_Mass_fixedAmp_002.txt   Condiciones iniciales normales

#aa = get_params_gal("/Users/Usuario/Documents/Master_Fisica_Teorica/TFM/unit_fixedAmp_002.10000000.txt",[31], 'UNIT', 'taurus', 1.041e-16, 10.5, 16.5, 60, 'resultados.h5', testing = False, verbose = True)
aa = get_params_gal("/home2/guillermo/data_grp/Halos_tree_DOC_PID_Vmax_all_Mass_fixedAmp_002.txt",[31], 'UNIT', 'taurus2', 1.041e-16, 10.5, 16.5, 60, 'resultadosred.h5', testing = False, verbose = True)

