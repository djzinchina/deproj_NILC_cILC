import numpy as np
import matplotlib
# matplotlib.use('Agg')
import healpy as hp 
import multiprocessing as mp 
import joblib as jl 

def get_disc(nside_super, vec, radius):
    disc_pix = hp.query_disc(nside_super, vec, radius, inclusive=True)
    return np.array(disc_pix)

def get_super_neighbours(nside_super, padded=False):
    n_cores = mp.cpu_count()

    super_pix_arr = np.arange(hp.nside2npix(nside_super))
    super_pix_vec = np.array(hp.pix2vec(nside_super, super_pix_arr))

    radius = 4.5* hp.nside2resol(nside_super)
    
    neighbour_list = jl.Parallel(n_jobs=n_cores)(jl.delayed(get_disc)(nside_super, super_pix_vec[:, ivec], radius) for ivec in super_pix_arr)

    # print(neighbour_list)

    if padded:
        size_list = []
        for arr in neighbour_list:
            size_list.append(len(arr))

        max_size = np.max(np.array(size_list))
        neighbour_array = - np.ones((hp.nside2npix(nside_super), max_size), dtype=np.int_) 

        for i in range(len(neighbour_list)):
            neighbour_array[i,:size_list[i]] = neighbour_list[i]
        
        return neighbour_array
    else:
        return neighbour_list

def npix2spix_map(nside_map, nside_super):
    
    npix_map = np.arange(hp.nside2npix(nside_map))

    x, y, z = hp.pix2vec(nside_map, npix_map)

    spix_map = hp.vec2pix(nside_super, x,y,z)

    return spix_map

def neighbour2index(npix_spix_map, neighbours):
    return np.where(np.in1d(npix_spix_map, np.array(neighbours)))


def get_map_indices(wvlt_nside, super_nside):

    npix_super = hp.nside2npix(super_nside)
    super_neighbours = get_super_neighbours(super_nside)
    if len(super_neighbours) != npix_super:
        print("Super lengths wrong!")

    wvlt_spix_map = npix2spix_map(wvlt_nside, super_nside)

    n_cores = mp.cpu_count()

    indices = jl.Parallel(n_jobs=n_cores)(jl.delayed(neighbour2index)(wvlt_spix_map, super_neighbours[i]) for i in range(npix_super))

    return indices

def compute_neighbour_array(nside_super, outfile_path=None):
    neighbour_array = get_super_neighbours(nside_super, padded=True)

    if outfile_path != None:
        np.savez_compressed(outfile_path+'/super_pix_neighbour_array_NSIDE-'+str(nside_super).zfill(4)+'.npz', neighbour_array=neighbour_array)
    
    return neighbour_array

def load_neighbour_array(nside_super, file_path):
    try:
        neighbour_array = np.load(file_path+'/super_pix_neighbour_array_NSIDE-'+str(nside_super).zfill(4)+'.npz')['neighbour_array']
    except:
        neighbour_array = compute_neighbour_array(nside_super, outfile_path=file_path)

    return neighbour_array

