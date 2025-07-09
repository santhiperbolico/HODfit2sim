import sys
import os
import numpy as np
import h5py
from src.h2s_profile_r import boundary_correction

def shuffle_parent_halos(
    input_hdf5,
    output_hdf5,
    id_field="id",
    pid_field="pid",
    rng_seed=None,
    verbose=True
):
    """
    Shuffle all properties of parent halos (pid == -1) within each mass bin, 
    storing both the original and shuffled sets in the output.

    Parameters
    ----------
    input_hdf5 : str
        Path to input HDF5 file (mass bins as groups).
    output_hdf5 : str
        Path to output HDF5 file.
    id_field : str
        Dataset name for halo ID.
    pid_field : str
        Dataset name for parent ID.
    rng_seed : int or None
        Seed for reproducibility.
    verbose : bool
        If True, print status messages.
    """
    rng = np.random.default_rng(rng_seed)
    with h5py.File(input_hdf5, "r") as f_in, h5py.File(output_hdf5, "w") as f_out:
        for group_name in f_in:
            grp_in = f_in[group_name]
            ids = grp_in[id_field][:]
            pids = grp_in[pid_field][:]
            parent_mask = (pids == -1)
            n_parents = parent_mask.sum()
            if n_parents == 0:
                if verbose:
                    print(f"Bin {group_name}: no parent halos found, skipping.")
                continue

            # Collect all property names (excluding id/pid)
            prop_fields = [key for key in grp_in.keys() if key not in [id_field, pid_field]]

            # Build original properties array
            original_dict = {}
            for key in [id_field, "X", "Y", "Z", "vx", "vy", "vz", "Mass"]:
                if key in grp_in:
                    original_dict[key] = grp_in[key][:][parent_mask]
                else:
                    raise ValueError(f"Expected field '{key}' not found in {group_name}")

            # Create a "pack" of all properties to shuffle in block
            # Shape: (n_parents, n_properties)
            n_properties = len(original_dict)
            original_matrix = np.column_stack([original_dict[k] for k in original_dict])

            # Shuffle indices
            indices = np.arange(n_parents)
            rng.shuffle(indices)

            # Shuffled matrix: same shape, shuffled along axis 0
            shuffled_matrix = original_matrix[indices, :]

            grp_out = f_out.create_group(group_name)
            # Save original and shuffled properties with clear names
            for i, key in enumerate(original_dict):
                grp_out.create_dataset(f"{key}_original", data=original_matrix[:, i])
                grp_out.create_dataset(f"{key}_shuffled", data=shuffled_matrix[:, i])

            # Para referencia: guarda también el mapping (índices)
            grp_out.create_dataset("shuffle_indices", data=indices)

            if verbose:
                print(f"Bin {group_name}: {n_parents} parent halos shuffled and written.")

    if verbose:
        print(f"Shuffled parent halos saved to: {output_hdf5}")

def compute_relative_offset(dx, boxsize):
    """
    Computes minimum image separation for periodic box (returns displacement between -L/2 and +L/2).
    """
    half_box = boxsize / 2.0
    dx_corr = np.copy(dx)
    dx_corr[dx_corr >  half_box] -= boxsize
    dx_corr[dx_corr < -half_box] += boxsize
    return dx_corr

def load_shuffled_halos_from_bins(h5file, verbose=True):
    """
    Load all shuffled parent halos from a halo_mass_bins_shuffled.h5 file.
    Returns a dictionary: id_shuffled -> (x, y, z, vx, vy, vz, mass)
    """
    id_shuffled_all = []
    x_all, y_all, z_all = [], [], []
    vx_all, vy_all, vz_all = [], [], []
    mass_all = []

    with h5py.File(h5file, "r") as f:
        for i in range(70):
            group = f[f"bin_{i:02d}"]
            id_shuffled_all.append(group["id_shuffled"][:])
            x_all.append(group["X"][:])
            y_all.append(group["Y"][:])
            z_all.append(group["Z"][:])
            vx_all.append(group["vx"][:])
            vy_all.append(group["vy"][:])
            vz_all.append(group["vz"][:])
            mass_all.append(group["Mass"][:])

    id_shuffled = np.concatenate(id_shuffled_all)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    z = np.concatenate(z_all)
    vx = np.concatenate(vx_all)
    vy = np.concatenate(vy_all)
    vz = np.concatenate(vz_all)
    mass = np.concatenate(mass_all)

    if verbose:
        print(f"Loaded {len(id_shuffled)} shuffled parent halos from 70 bins.")

    # Diccionario id_shuffled → (x, y, z, vx, vy, vz, mass)
    halo_dict = {hid: (x_, y_, z_, vx_, vy_, vz_, m_)
                 for hid, x_, y_, z_, vx_, vy_, vz_, m_ in zip(id_shuffled, x, y, z, vx, vy, vz, mass)}
    return halo_dict

def shuffle_galaxy_catalog_binned(
    galaxy_file,
    halo_shuffled_file,
    output_file,
    boxsize=1000.0,
    bins=70,
    galaxy_id_field="MainHaloID",
    galaxy_host_field="HostHaloID",
    galaxy_x_field="Xpos",
    galaxy_y_field="Ypos",
    galaxy_z_field="Zpos",
    galaxy_vx_field="Xvel",
    galaxy_vy_field="Yvel",
    galaxy_vz_field="Zvel",
    halo_id_original_field="id_original",
    halo_id_shuffled_field="id_shuffled",
    halo_x_original_field="X_original",
    halo_y_original_field="Y_original",
    halo_z_original_field="Z_original",
    halo_vx_original_field="vx_original",
    halo_vy_original_field="vy_original",
    halo_vz_original_field="vz_original",
    halo_x_shuffled_field="X_shuffled",
    halo_y_shuffled_field="Y_shuffled",
    halo_z_shuffled_field="Z_shuffled",
    halo_vx_shuffled_field="vx_shuffled",
    halo_vy_shuffled_field="vy_shuffled",
    halo_vz_shuffled_field="vz_shuffled",
    verbose=True,
    boundary_correction=None,  
):
    """
    Shuffle galaxies: each galaxy is assigned a new parent halo (from the same mass bin), and its position/velocity is
    re-centered relative to the new halo, preserving the offset wrt the original halo.
    Applies periodic boundary correction to final positions.
    """

    import h5py
    import numpy as np

    with h5py.File(galaxy_file, "r") as f:
        host_id = f[galaxy_host_field][:]
        main_id = f[galaxy_id_field][:]
        xpos = f[galaxy_x_field][:]
        ypos = f[galaxy_y_field][:]
        zpos = f[galaxy_z_field][:]
        vx = f[galaxy_vx_field][:]
        vy = f[galaxy_vy_field][:]
        vz = f[galaxy_vz_field][:]
        other_fields = {k: f[k][:] for k in f if k not in [
            galaxy_host_field, galaxy_id_field, galaxy_x_field, galaxy_y_field, galaxy_z_field,
            galaxy_vx_field, galaxy_vy_field, galaxy_vz_field
        ]}

    N = len(xpos)
    if verbose:
        print(f"Loaded {N} galaxies.")

    # Prepare output arrays
    x_new, y_new, z_new = np.copy(xpos), np.copy(ypos), np.copy(zpos)
    vx_new, vy_new, vz_new = np.copy(vx), np.copy(vy), np.copy(vz)
    main_id_new = np.copy(main_id)
    host_id_new = np.copy(host_id)  # Optional, but keep in sync

    # For each mass bin...
    with h5py.File(halo_shuffled_file, "r") as f_halo:
        for i in range(bins):
            gname = f"bin_{i:02d}"
            if gname not in f_halo:
                continue
            g = f_halo[gname]
            id_original = g[halo_id_original_field][:]
            id_shuffled = g[halo_id_shuffled_field][:]
            X_ori = g[halo_x_original_field][:]
            Y_ori = g[halo_y_original_field][:]
            Z_ori = g[halo_z_original_field][:]
            vx_ori = g[halo_vx_original_field][:]
            vy_ori = g[halo_vy_original_field][:]
            vz_ori = g[halo_vz_original_field][:]
            X_shuf = g[halo_x_shuffled_field][:]
            Y_shuf = g[halo_y_shuffled_field][:]
            Z_shuf = g[halo_z_shuffled_field][:]
            vx_shuf = g[halo_vx_shuffled_field][:]
            vy_shuf = g[halo_vy_shuffled_field][:]
            vz_shuf = g[halo_vz_shuffled_field][:]

            n_halos = len(id_original)
            if n_halos == 0:
                continue

            # Todas las galaxias cuyo main_id estaba en id_original de este bin
            idx_gal = np.where(np.isin(main_id, id_original))[0]
            if len(idx_gal) == 0:
                continue

            # Creamos un mapeo id_original --> idx dentro del bin
            id_to_idx = {hid: idx for idx, hid in enumerate(id_original)}

            for gal_idx in idx_gal:
                gal_main_id = main_id[gal_idx]
                is_central = (host_id[gal_idx] == main_id[gal_idx])

                if gal_main_id not in id_to_idx:
                    continue  # safety

                orig_idx = id_to_idx[gal_main_id]
                # Find offset to original halo (aplicar condiciones periódicas)
                dx = xpos[gal_idx] - X_ori[orig_idx]
                dy = ypos[gal_idx] - Y_ori[orig_idx]
                dz = zpos[gal_idx] - Z_ori[orig_idx]


                # Asignamos la galaxia al halo shuffleado (misma posición dentro del bin)
                x_new[gal_idx] = X_shuf[orig_idx] + dx
                y_new[gal_idx] = Y_shuf[orig_idx] + dy
                z_new[gal_idx] = Z_shuf[orig_idx] + dz

                # OJO: los offsets hay que corregirlos periódicamente para que sean [-L/2, L/2]
                if boundary_correction is not None:
                    x_new = boundary_correction(x_new, boxsize, groups=True)
                    y_new = boundary_correction(y_new, boxsize, groups=True)
                    z_new = boundary_correction(z_new, boxsize, groups=True)

                vx_new[gal_idx] = vx_shuf[orig_idx] + (vx[gal_idx] - vx_ori[orig_idx])
                vy_new[gal_idx] = vy_shuf[orig_idx] + (vy[gal_idx] - vy_ori[orig_idx])
                vz_new[gal_idx] = vz_shuf[orig_idx] + (vz[gal_idx] - vz_ori[orig_idx])

                main_id_new[gal_idx] = id_shuffled[orig_idx]
                # host_id no cambia

            if verbose:
                print(f"Shuffled bin {i:02d}: {len(idx_gal)} galaxies, {n_halos} halos")

    # --- Corrige las posiciones finales para estar dentro de la caja ---
    if boundary_correction is not None:
        x_new = boundary_correction(x_new, boxsize, groups=False)
        y_new = boundary_correction(y_new, boxsize, groups=False)
        z_new = boundary_correction(z_new, boxsize, groups=False)
        if verbose:
            print("Applied periodic boundary correction to final positions.")

    # 3. Save output
    if verbose:
        print(f"Saving shuffled catalog: {output_file}")

    with h5py.File(output_file, "w") as fout:
        fout.create_dataset(galaxy_host_field, data=host_id_new)
        fout.create_dataset(galaxy_id_field, data=main_id_new)
        fout.create_dataset(galaxy_x_field, data=x_new)
        fout.create_dataset(galaxy_y_field, data=y_new)
        fout.create_dataset(galaxy_z_field, data=z_new)
        fout.create_dataset(galaxy_vx_field, data=vx_new)
        fout.create_dataset(galaxy_vy_field, data=vy_new)
        fout.create_dataset(galaxy_vz_field, data=vz_new)
        for k, v in other_fields.items():
            fout.create_dataset(k, data=v)

    if verbose:
        print("Done.")
