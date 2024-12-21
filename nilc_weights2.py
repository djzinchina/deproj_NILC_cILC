import numpy as np
import healpy as hp
import super_pix2 as sp 
import covarifast as co

class nilc_weights_new:
    '''
        New workspace class for NILC weights calculation workspace.
        This class has lot of memory requirements making it unsuitable for smaller systems.
    '''
    def __init__(self, wvlt_maps, super_nside=None):
        possible_wav_nside = [16,32,64,128,256,512,1024,2048,4096,8192]
        correspo_sup_nside = [1,1,2,2,4,4,8,8,16,16]

        self.nu_dim = len(wvlt_maps)
        self.__wav_maps = np.array(wvlt_maps)

        wvlt_nside = hp.npix2nside(len(wvlt_maps[0]))

        if super_nside == None:
            if wvlt_nside in possible_wav_nside :
                super_nside = correspo_sup_nside[np.where(np.array(possible_wav_nside) == wvlt_nside)[0][0]]
            else :
                print('Case not implemented')

        self.nside_map = wvlt_nside
        self.nside_sup = super_nside

        self.__spix_groups_arr = sp.load_neighbour_array(super_nside, '/media/doujzh/AliCPT_data/NILC_neighbours')

        self.__s2nind = sp.npix2spix_map(self.nside_map, self.nside_sup)

    def get_weights(self):
        e_vec = np.ones((self.nu_dim,), dtype=np.float64)

        npix_sup = hp.nside2npix(self.nside_sup)
        npix_map = hp.nside2npix(self.nside_map)
        
        covmat_map = co.compute_cov_mat(self.__spix_groups_arr, self.__s2nind, self.__wav_maps)

        # self.cov_map = covmat_map
        covinv_map = np.linalg.pinv(covmat_map)

        del covmat_map

        weights_smap = np.matmul(e_vec, covinv_map) / np.matmul(e_vec, np.matmul(covinv_map, e_vec.T).reshape(npix_sup,self.nu_dim,1))

        # print(self.nu_dim, np.array(weights_smap).shape)
        del covinv_map

        weights_wav = np.zeros((self.nu_dim, hp.nside2npix(self.nside_map),))

        # print(weights_wav.shape, w2s_map.shape)

        for spix in range(npix_sup) :
            # print(np.where(w2s_map == spix)[0].shape)
            # hp.mollview(w2s_map[np.where(w2s_map == spix)[0]], cmap=plt.cm.plasma)
            # plt.show()
            # print(weights_smap[spix].shape)
            pix_map = list(np.where(self.__s2nind == spix)[0])
            # print(weights_wav[:,pix_map].shape)
            weights_wav[:,pix_map] = weights_smap[spix].reshape(self.nu_dim,1)*np.ones((self.nu_dim,len(pix_map)))

        del weights_smap

        return weights_wav