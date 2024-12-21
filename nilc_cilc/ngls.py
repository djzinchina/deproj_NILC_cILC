'''
Module for Generalized Least Square Method in Needlet Space
'''

import numpy as np
import healpy as hp
from tqdm import tqdm
import shutil
import random, string
from pathlib import Path
import matplotlib.pyplot as plt

import cmbframe.plot_utilts as pu
import cmbframe.cleaning_utils as cu
import cmbframe.hp_wrapper as hw 
import cmbframe.wavelet as wv
import covar as co
import super_pix2 as su

input_map_prefix = '/map_in_wav-transform-'
proj_map_prefix = '/comp_in_wav-transform-'

def ugrade_wgt(weights, nside_up):
        nside_low = hp.npix2nside(len(weights[0,:]))
        n_valid_ch = len(weights[:,0])
        s2nind = su.npix2spix_map(nside_up, nside_low)

        nilc_wgts_mapped = np.zeros((n_valid_ch, hp.nside2npix(nside_up)))
        nilc_wgts_mapped = weights[:, s2nind]

        return nilc_wgts_mapped

def compute_Ncov(Nlms_in, bands, lmax_ch, beams, com_res_beam, cov_method='supix', ilc_bias=0.03):
    nu_dim = len(Nlms_in)
    nu_arr = np.arange(nu_dim, dtype=np.int_)
    beam_factor = cu.beam_ratios(beams, com_res_beam)
    nbands = len(bands[0,:])

    wav_maps = []
    for nu in range(nu_dim):
        Nlm_nu = hp.almxfl(np.copy(Nlms_in[nu]), beam_factor[nu])
        wav_maps_nu = wv.alm2wavelet(Nlm_nu, bands)
        wav_maps.append(wav_maps_nu)
        del wav_maps_nu

    Ncov_by_bands = []
    for band in range(nbands):
        lmax_band = wv.get_lmax_band(bands[:,band])
        map_nus_in_band = []
        for nu in nu_arr[lmax_ch >= lmax_band]:
            map_nus_in_band.append(wav_maps[nu][band])

        if cov_method == 'supix':
            Ncov = co.supix_covar(map_nus_in_band)
        elif cov_method == 'dgrade':
            Ncov = co.dgrade_covar(map_nus_in_band, bands[band], ilc_bias=ilc_bias)
        else:
            print("ERROR: cov_method value is not set to a valid method.")
        Ncov_by_bands.append(Ncov)


    del wav_maps, Nlms_in, Ncov
    return Ncov_by_bands

class ngls_workspace:
    """
    Class to setup NGLS cleaning for multifrequency CMB maps.

    ...

    Attributes
    ----------
    nside : int
        Healpix \(N_{\\rm side}\) parameter at which output map is required.
    maps_in : list of numpy ndarry
        A list of ``nu_dim`` healpy maps. Depending on whether IQU maps or scalar map, 
        each map is either a numpy 2D or 1D array. If IQU map is supplied 
        please set ``TEB`` to indicate which map is to be analysed.
    bands : numpy ndarray
        A numpy 2D array of shape ``(lmax+1, nbands)`` containing the needlet 
        bands. Here ``nbands`` are the number of needlet bands.
    beams : numpy ndarray
        A numpy 2D array  of shape ``(lmax+1, nu_dim)`` containing beams for 
        each of the ``nu_dim`` channels.
    com_res_beam : numpy array
        A numpy 1D array of shape ``(lmax+1)`` containing the beam for the common
        resolution beam smoothing.
    TEB : {0,1,2}, optional
        Selects which of TEB is to be analysed in case of IQU map input. Default is 0.
    mask : numpy array, optional
        A numpy 1d array containing the binary mask required for SHT of partial sky
        maps. Note that nilc weights do not consider masking. Default is ``None``, which
        does SHT without masking.
    lmax_sht : int, optional
        Specifies the \(\\ell_{\\rm max}\) for SHT of maps. It is suggested to set this 
        to ``lmax`` of the needlet bands. Default is ``None`` in which case ``lmax_sht``
        is the ``lmax`` inferred from the shape of the needlet bands array.
    low_memory : bool, optional
        If ``True`` the wavelet maps are not retained on RAM and are written to a scratch
        directory specified by ``scratch``. Default is ``True``.
    scratch : str, optional
        Sets the path of the scratch folder where temporary files are stored. This is not 
        used if ``low_memory=False``.
    wav_nside_set : list
        List of wavelet transform nside settings for `cmbframe.wavelet.alm2wavelet` function.
        For any required parameters you need to provide a list ``[nodegrade, wavelet_nside, 
        w_nside_min, w_nside_max]``. Refer to `cmbframe.wavelet.alm2wavelet` for details.

    """

    def __init__(self, nside, maps_in, bands, beams, com_res_beam, TEB=0, mask=None, lmax_sht=None, low_memory=False, scratch='./scratch', wav_nside_set=None):
        # maps_in = np.array(maps_in)
        self.nu_dim = len(maps_in)
        self.nbands = len(bands[0,:])
        self.nside = nside
        self.bands = bands
        isIQU = len(np.array(maps_in[0]).shape)

        if isinstance(mask, np.ndarray):
            self.mask = mask
        else:
            self.mask = np.ones((hp.nside2npix(self.nside),))
        self.low_memory = low_memory

        if self.low_memory:
            self.scratch = scratch
            Path(self.scratch).mkdir(parents=True, exist_ok=True)

        if lmax_sht == None:
            self.lmax_sht = len(bands[:,0])
        else:
            self.lmax_sht = lmax_sht

        if len(beams) != self.nu_dim:
            print('ERROR: Number of maps do not match number of beams. Aborting...')
            exit()

        if isinstance(wav_nside_set, list):
            self.nodegrade = wav_nside_set[0]
            self.wavelet_nside = wav_nside_set[1]
            self.w_nside_min = wav_nside_set[2]
            self.w_nside_max = wav_nside_set[3]

        self.env_id = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        self.beam_factor = cu.beam_ratios(beams, com_res_beam)

        for nu in tqdm(range(self.nu_dim), ncols=120):
            nside_mask = hp.npix2nside(len(self.mask))
            if isIQU == 1:
                nside_nu = hp.npix2nside(len(maps_in[nu]))
                if nside_nu != nside_mask:
                    mask_nu = hw.mask_udgrade(self.mask,nside_nu)
                else:
                    mask_nu = self.mask
                alm_nu = hp.map2alm(maps_in[nu]*mask_nu, lmax=self.lmax_sht, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
            elif isIQU == 2:
                nside_nu = hp.npix2nside(len(maps_in[nu][0]))
                if nside_nu != nside_mask:
                    mask_nu = hw.mask_udgrade(self.mask,nside_nu)
                else:
                    mask_nu = self.mask
                alm_nu = hp.map2alm(maps_in[nu]*mask_nu, lmax=self.lmax_sht, pol=True, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')[TEB]

            alm_nu = hp.almxfl(alm_nu, self.beam_factor[nu])

            try:
                wav_maps_nu = wv.alm2wavelet(alm_nu, self.bands, nodegrade=self.nodegrade, wavelet_nside=self.wavelet_nside, w_nside_min=self.w_nside_min, w_nside_max=self.w_nside_max)
            except AttributeError:
                wav_maps_nu = wv.alm2wavelet(alm_nu, self.bands)
            del alm_nu

            if self.low_memory:
                wv.write_waveletmap_fits(self.bands, wav_maps_nu, outfile=self.scratch+input_map_prefix+self.env_id+'_nu'+str(nu)+'.fits')
            else:
                if nu == 0:
                    self.wavelet_maps = [wav_maps_nu]
                else:
                    self.wavelet_maps.append(wav_maps_nu)

            del wav_maps_nu

    def compute_ngls_weights(self, Ncov, ch=[0,1,2,3,4,5,6], constr_comp=[1,2], bandpass='real', lmax_ch=None, return_clean_map=True):
        """
        Computes NGLS weights and optionally returns the NGLS cleaned map.

        len(lmax_ch) = len(ch) = self.nu_dim

        ch must be in the same sequence as the input maps_in_nus and lmax_ch

        Parameters
        ----------

        """
                              # CMB       # Sync      # Dust
        __A_real = np.array([[1.0000000,  109.44489, 0.054501805],            # WMAP K
                             [1.0000000,  1.8755465,  0.62621540],            # AliCPT 95
                             [1.0000000,  1.5965695,  0.70261379],            # HFI 100
                             [1.0000000, 0.73042459,   1.4810256],            # HFI 143
                             [1.0000000, 0.65970442,   1.6505071],            # AliCPT 150
                             [1.0000000, 0.37031326,   5.0996852],            # HFI 217
                             [1.0000000, 0.38296325,   40.634704]])           # HFI 343

                               # CMB       # Sync      # Dust
        __A_delta = np.array([[1.0000000,  105.40067, 0.054502588],           # WMAP K
                              [1.0000000,  1.8532817,  0.59631832],           # AliCPT 95
                              [1.0000000,  1.6276116,  0.65881741],           # HFI 100
                              [1.0000000, 0.71536776,   1.4191725],           # HFI 143
                              [1.0000000, 0.65023833,   1.5924489],           # AliCPT 150
                              [1.0000000, 0.37032728,   4.5349128],           # HFI 217
                              [1.0000000, 0.37115226,   35.319678]])          # HFI 343
        ch = np.array(ch)
        nu_arr = np.arange(self.nu_dim, dtype=np.int_)
        comp = np.insert(constr_comp,0, 0, axis=0)
        comp = comp.astype(int)
        e_vec = np.zeros((len(comp),))
        e_vec[0] = 1. 

        if np.logical_not(isinstance(lmax_ch, (list, np.ndarray))):
            self.lmax_ch = self.lmax_sht * np.ones_like(np.array(nu_arr))
        else:
            self.lmax_ch = lmax_ch

        nilc_wavelet_maps = []
        for band in tqdm(range(self.nbands), ncols=120):

            lmax_band = wv.get_lmax_band(self.bands[:,band])

            map_sel = nu_arr[self.lmax_ch >= lmax_band]
            ch_sel = ch[self.lmax_ch >= lmax_band]

            if bandpass == 'real':
                A_part = np.copy(__A_real[ch_sel])[:,comp]
            elif bandpass == 'delta':
                A_part = np.copy(__A_delta[ch_sel])[:,comp]
            A_part = np.reshape(A_part, (len(ch_sel), len(comp)))


            map_nus_in_band = []
            for nu in map_sel:
                if self.low_memory:
                    map_nus_in_band.append(wv.read_waveletmap_fits(self.scratch+input_map_prefix+self.env_id+'_nu'+str(nu)+'.fits', band_no=band))
                else:
                    map_nus_in_band.append(self.wavelet_maps[nu][band])

            n_valid_ch = len(map_sel)

            # npix_low =  len(Ncov[band][:,0,0])
            cov_inv = np.linalg.pinv(Ncov[band])
            # det = np.linalg.det(cov_inv)
            # if np.any(det == 0.) or np.any(np.isnan(det)):
            #     print("det Zero/NaN found")
            #     for i in range(npix_low):
            #         if det[i] == 0.:
            #             print("det Zero for",i)
            #             print(cov_inv[i])
            #             exit()
            #         if np.isnan(det[i]):
            #             print("det NaN for", i)            
            # weights = np.matmul(e_vec_band, cov_inv)
            # print(cov_inv.shape) 
            # print(weights.shape, npix_low,n_valid_ch)
            # test = np.matmul(cov_inv, e_vec_band.T)#.reshape(npix_low,n_valid_ch,1)
            # print(test.shape)
            wgts_part = np.matmul(A_part.T, np.copy(cov_inv))
            At_Cinv_A_inv = np.linalg.inv(np.matmul(A_part.T, np.matmul(np.copy(cov_inv), A_part)))
            weights = np.matmul(At_Cinv_A_inv, wgts_part)
            # weights.shape=(npix_low, n_comp, n_valid_ch)
            weights = np.swapaxes(np.array(weights), 0, 1)
            weights = np.swapaxes(weights, 1, 2)
            # weights.shape=(n_comp, n_valid_ch, npix_low)

            # print(weights.shape)

            if self.low_memory:
                np.save(self.scratch+'/nilc_weights_env-'+self.env_id+'_band'+str(band)+'.npy', weights)
            else:
                if band == 0:
                    self.weights_by_band = []
                    
                self.weights_by_band.append(weights)
            
            if return_clean_map:
                wgts_by_nu = ugrade_wgt(weights[0], hp.npix2nside(len(map_nus_in_band[0])))
                for nu in range(n_valid_ch):
                    if nu == 0:
                        cleaned_wav = wgts_by_nu[nu] * map_nus_in_band[nu]
                    else:
                        cleaned_wav += wgts_by_nu[nu] * map_nus_in_band[nu]
                
                del wgts_by_nu, map_nus_in_band

                nilc_wavelet_maps.append(cleaned_wav)

        self.weights_done = True 

        if return_clean_map:
            return wv.wavelet2map(self.nside, nilc_wavelet_maps, self.bands) * self.mask


    def get_ngls_maps(self, comp=0):
        if self.weights_done :
            
            nu_arr = np.arange(self.nu_dim, dtype=np.int_)

            nilc_wavelet_maps = []
            for band in range(self.nbands):
                lmax_band = wv.get_lmax_band(self.bands[:,band])
                map_nus_in_band = []
                for nu in nu_arr[self.lmax_ch >= lmax_band]:
                    if self.low_memory:
                        map_nus_in_band.append(wv.read_waveletmap_fits(self.scratch+input_map_prefix+self.env_id+'_nu'+str(nu)+'.fits', band_no=band))
                    else:
                        map_nus_in_band.append(self.wavelet_maps[nu][band])

                n_valid_ch = len(nu_arr[self.lmax_ch >= lmax_band])

                if self.low_memory:
                    weights = np.load(self.scratch+'/nilc_weights_env-'+self.env_id+'_band'+str(band)+'.npy')
                else:
                    weights = self.weights_by_band[band]

                wgts_by_nu = ugrade_wgt(weights[comp], hp.npix2nside(len(map_nus_in_band[0])))

                del weights

                for nu in range(n_valid_ch):
                    if nu == 0:
                        cleaned_wav = wgts_by_nu[nu] * map_nus_in_band[nu]
                    else:
                        cleaned_wav += wgts_by_nu[nu] * map_nus_in_band[nu]

                nilc_wavelet_maps.append(cleaned_wav)

                del wgts_by_nu, map_nus_in_band, cleaned_wav
            
            return wv.wavelet2map(self.nside, nilc_wavelet_maps, self.bands) * self.mask

        else:
            print("ERROR: NILC weights not computed. Aborting...")
            exit()

    def get_projected_map(self, maps_in, TEB=0, adjust_beam=True, comp=0):
        # maps_in = np.array(maps_in)
        isIQU = len(np.array(maps_in[0]).shape)

        for nu in tqdm(range(self.nu_dim), ncols=120):
            nside_mask = hp.npix2nside(len(self.mask))
            if isIQU == 1:
                nside_nu = hp.npix2nside(len(maps_in[nu]))
                if nside_nu != nside_mask:
                    mask_nu = hw.mask_udgrade(self.mask,nside_nu)
                else:
                    mask_nu = self.mask
                alm_nu = hp.map2alm(maps_in[nu] * mask_nu, lmax=self.lmax_sht, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
            elif isIQU == 2:
                nside_nu = hp.npix2nside(len(maps_in[nu][0]))
                if nside_nu != nside_mask:
                    mask_nu = hw.mask_udgrade(self.mask,nside_nu)
                else:
                    mask_nu = self.mask
                alm_nu = hp.map2alm(maps_in[nu] * mask_nu, lmax=self.lmax_sht, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')[TEB]

            if adjust_beam:
                alm_nu = hp.almxfl(alm_nu, self.beam_factor[nu])

            try:
                wav_maps_nu = wv.alm2wavelet(alm_nu, self.bands, nodegrade=self.nodegrade, wavelet_nside=self.wavelet_nside, w_nside_min=self.w_nside_min, w_nside_max=self.w_nside_max)
            except AttributeError:
                wav_maps_nu = wv.alm2wavelet(alm_nu, self.bands)
            del alm_nu

            if self.low_memory:
                wv.write_waveletmap_fits(self.bands, wav_maps_nu, outfile=self.scratch+proj_map_prefix+self.env_id+'_nu'+str(nu)+'.fits')
    
            else:
                if nu == 0:
                    proj_wavelet_maps = [wav_maps_nu]
                else:
                    proj_wavelet_maps.append(wav_maps_nu)
                
            del wav_maps_nu

        if self.weights_done :
            
            nu_arr = np.arange(self.nu_dim, dtype=np.int_)

            nilc_wavelet_proj = []
            for band in range(self.nbands):
                lmax_band = wv.get_lmax_band(self.bands[:,band])

                map_nus_in_band = []
                for nu in nu_arr[self.lmax_ch >= lmax_band]:
                    if self.low_memory:
                        map_nus_in_band.append(wv.read_waveletmap_fits(self.scratch+proj_map_prefix+self.env_id+'_nu'+str(nu)+'.fits', band_no=band))
                    else:
                        map_nus_in_band.append(proj_wavelet_maps[nu][band])

                n_valid_ch = len(nu_arr[self.lmax_ch >= lmax_band])

                if self.low_memory:
                    weights = np.load(self.scratch+'/nilc_weights_env-'+self.env_id+'_band'+str(band)+'.npy')
                else:
                    weights = self.weights_by_band[band]

                wgts_by_nu = ugrade_wgt(weights[comp], hp.npix2nside(len(map_nus_in_band[0])))

                del weights

                for nu in range(n_valid_ch):
                    if nu == 0:
                        cleaned_wav = wgts_by_nu[nu] * map_nus_in_band[nu]
                    else:
                        cleaned_wav += wgts_by_nu[nu] * map_nus_in_band[nu]

                nilc_wavelet_proj.append(cleaned_wav)

                del wgts_by_nu, map_nus_in_band, cleaned_wav

            if np.logical_not(self.low_memory):
                del proj_wavelet_maps
            
            return wv.wavelet2map(self.nside, nilc_wavelet_proj, self.bands) * self.mask

        else:
            print("ERROR: NILC weights not computed. Aborting...")
            exit()



    def clean(self):
        if self.low_memory:
            print("Deleting scratch folder. All computation files will be lost. Workspace will not function after cleaning.")
            shutil.rmtree(self.scratch, ignore_errors=True)
        else:
            print("Nothing to do. Not using scratch.")

    
    def plot_nilc_weights(self, freqs, comp=0, mask=None, wgt_type='CMB', label_list=None, outfile=None, resol='screen', show=True):
        if self.weights_done:
            nu_arr = np.arange(self.nu_dim, dtype=np.int_)

            if isinstance(label_list,(list, np.ndarray)):
                use_label = label_list
            else : 
                use_label = list('band '+str(band) for band in range(self.nbands))

            if isinstance(mask, np.ndarray):
                mask_in = mask
            else : 
                mask_in = self.mask

            fno, fig, ax = pu.make_plotaxes(res=resol, shape='rec')

            for band in range(self.nbands):
                lmax_band = wv.get_lmax_band(self.bands[:,band])

                if self.low_memory:
                    weights = np.load(self.scratch+'/nilc_weights_env-'+self.env_id+'_band'+str(band)+'.npy')
                else:
                    weights = self.weights_by_band[band]
                wgts_by_nu = ugrade_wgt(weights[comp], hp.npix2nside(len(mask_in)))
                avg_wgt_by_nu = np.mean(np.multiply(wgts_by_nu, mask_in), axis=1)
                ax.plot(nu_arr[self.lmax_ch >= lmax_band], avg_wgt_by_nu, 'o-', label=use_label[band])
            ax.set_xlabel(r'Frequency[GHz]')
            ax.set_ylabel(r'Mean weight of '+wgt_type)
            ax.set_xticks(nu_arr, np.array(freqs))
            if resol =='screen': ax.legend(loc='best', frameon=False, fontsize=12)
            if resol =='print': ax.legend(loc='best', frameon=False, fontsize=8)
            # ax.set_xscale("log")
            if outfile != None : plt.savefig(outfile,bbox_inches='tight',pad_inches=0.1)
            if show: plt.show()
        else:
            print("ERROR: NILC weights not computed. Aborting...")
            exit()            
         


            

