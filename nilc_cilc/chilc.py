"""
Module for harmonic domain contrained ILC foreground cleaning method for CMB B-modes.
"""
import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt  

import cmbframe.cleaning_utils as cu

def calc_binned_cov(alm1, alm2=None, bsz=0.4, excld_centr_mode=True):
    '''
    Range to compute covariance matrix is from min(0.6*ell, ell-7) to
    max(1.4*ell, ell+7) to avoid too large ILC bias
    '''
    ALM = hp.Alm()
    lmax = ALM.getlmax(len(alm1))

    Cl_1x2 = hp.alm2cl(alm1, alms2=alm2)

    ells = np.arange(lmax+1)
    mode_factor = 2.*ells + 1. 
    Cl_1x2 = mode_factor * Cl_1x2

    Cl_binned = np.zeros((lmax+1,))

    for li in range(2, len(Cl_1x2)) :
        limin = np.maximum(int(np.floor(np.minimum((1-bsz)*li, li-7))), 2)
        limax = np.minimum(int(np.ceil(np.maximum((1+bsz)*li, li+7))), lmax-1)
        if excld_centr_mode:
            Cl_binned[li] = (np.sum(Cl_1x2[limin:limax]) - Cl_1x2[li]) / np.sum(mode_factor[limin:limax]) #(limax - limin) 
        else:
            Cl_binned[li] = np.sum(Cl_1x2[limin:limax]) / np.sum(mode_factor[limin:limax]) #(limax - limin) 

    del Cl_1x2 
    return Cl_binned
    
class cilc_cleaner:
    """
    This class defines the workspace for harmonic domain constrained ILC method.
    
    ...

    Attributes
    ----------
    alms_in : complex numpy ndarray  
        Numpy 2d array of shape ``(nu_dim, len(alm))`` containing healpix spherical 
        harmonic coefficients for ``nu_dim`` maps for different frequency channels.
    beams : numpy ndarray
        A numpy 2D array of shape ``(nu_dim, lmax+1)``. This contains beams for nu_dim channels. 
        If polarized beam, then it contains either the E component or the B component depending 
        on which map/alm is being targeted. This represents \(B^{T/E/B}_{\\ell}\) for the 
        different maps in the set.
    com_res_beam : numpy array
        A numpy 1D array of shape ``(lmax+1)`` representing the beam of the common resolution 
        that is being targetted.

    Methods
    -------
    compute_cilc_weights(map_sel, constr_comp=[1,2], bandpass='delta', lmax_ch=None)
        Computes the cILC weights for cleaning.
    get_cleaned_alms()
        Returns cleaned alm.
    get_projected_alms(alms_to_proj, adjust_beam=True)
        Returns alms projected with the cILC weights.
    """

    def __init__(self, alms_in, beams=None, com_res_beam=None):

        ALM = hp.Alm()
        self.lmax = ALM.getlmax(len(alms_in[0,:]))
        self.nu_dim = len(alms_in[:,0])

        if isinstance(beams, (tuple,np.ndarray)) and isinstance(com_res_beam, np.ndarray):
            self.beam_ratio = cu.beam_ratios(beams, com_res_beam)

        self.alms = []
        for nu in range(self.nu_dim):
            self.alms.append(hp.almxfl(np.copy(alms_in[nu]), self.beam_ratio[nu]))

        self.alms = np.array(self.alms)

    def compute_cilc_weights(self, map_sel, constr_comp=[1,2], bandpass='delta', lmax_ch=None, bsz=0.4, excld_centr_mode=True):
        """
        Computes cILC weights with constraints on average dust and/or synchrotron emissions.

        Parameters
        ----------
        map_sel : list or numpy array of int
            A list or numpy 1D array of channels from ``0 = WMAP K, 1 = AliCPT 95, 2 = HFI 100 
            3 = HFI 143, 4 = AliCPT 150, 5 = HFI 217, 6 = HFI 353``.
        constr_comp : list of int, optional
            Emission components to constrain from 1 = dust emissions, 2 = synchrotron emissions.
            Default is [1,2] to constrain both dust and synchrotron.
        bandpass : {'delta', 'real'}, optional
            Chooses how the mixing matrix is computed. If option is ``'delta'``  then we delta 
            bandpass while ``'real'``uses realistic bandpass. Default is ``'delta'``.
        lmax_ch : list or numpy array, optional
            Sets the list of ``lmax`` for every included frequency channels. If not set then 
            all channels have the same maximum ``lmax``. Default is ``None``.
        """
                              # CMB       # Dust      # Sync
        __A_real = np.array([[1.0000000, 0.054501805,  109.44489],            # WMAP K
                             [1.0000000,  0.62621540,  1.8755465],            # AliCPT 95
                             [1.0000000,  0.70261379,  1.5965695],            # HFI 100
                             [1.0000000,   1.4810256, 0.73042459],            # HFI 143
                             [1.0000000,   1.6505071, 0.65970442],            # AliCPT 150
                             [1.0000000,   5.0996852, 0.37031326],            # HFI 217
                             [1.0000000,   40.634704, 0.38296325]])           # HFI 343

                               # CMB       # Dust      # Sync
        __A_delta = np.array([[1.0000000, 0.054502588,  105.40067],           # WMAP K
                              [1.0000000,  0.59631832,  1.8532817],           # AliCPT 95
                              [1.0000000,  0.65881741,  1.6276116],           # HFI 100
                              [1.0000000,   1.4191725, 0.71536776],           # HFI 143
                              [1.0000000,   1.5924489, 0.65023833],           # AliCPT 150
                              [1.0000000,   4.5349128, 0.37032728],           # HFI 217
                              [1.0000000,   35.319678, 0.37115226]])          # HFI 343

        if bandpass == 'real':
            self.A = np.copy(__A_real[map_sel])
        elif bandpass == 'delta':
            self.A = np.copy(__A_delta[map_sel])

        comp = np.insert(constr_comp,0, 0, axis=0)
        # print(comp)

        self.A = self.A[:,comp]

        har_cov_ij = np.zeros((self.lmax-1,self.nu_dim,self.nu_dim))

        for nu_1 in range(0,self.nu_dim) :
            for nu_2 in range(nu_1,self.nu_dim) :
                if nu_2 == nu_1:
                    har_cov_ij[:,nu_1,nu_1] = calc_binned_cov(self.alms[nu_1], bsz=bsz, excld_centr_mode=excld_centr_mode)[2:]
                else:
                    har_cov_ij[:,nu_1,nu_2] = har_cov_ij[:,nu_2,nu_1] = calc_binned_cov(self.alms[nu_1], alm2=self.alms[nu_2], bsz=bsz, excld_centr_mode=excld_centr_mode)[2:]

        # print(har_cov_ij.shape)
        e_vec = np.zeros((len(comp),))
        e_vec[0] = 1. 
        e_vec = np.reshape(e_vec, (len(comp),1))

        if np.logical_not(isinstance(lmax_ch, (list, np.ndarray))):
            lmax_ch = self.lmax * np.ones_like(np.array(map_sel))

        nu_arr = np.arange(self.nu_dim, dtype=np.int32)

        self.har_wgts = np.zeros((self.nu_dim, self.lmax+1))
        
        lmax_ch_uni = np.unique(lmax_ch)

        for ell in lmax_ch_uni:
            ell_wh = np.where(lmax_ch_uni == ell)[0][0]
            # print(ell_wh)
            if ell_wh == 0:
                ell_min_slice = 2
            else:
                ell_min_slice = lmax_ch_uni[ell_wh - 1]
            # print(ell_min_slice)
            nu_sel = nu_arr[lmax_ch >= ell]
            # print(nu_sel)
            sliced_har_cov_ij = np.copy(har_cov_ij[ell_min_slice-2:ell-1,:,:][:,nu_sel,:][:,:,nu_sel])#
            A_part = np.copy(self.A[nu_sel,:])

            # print(sliced_har_cov_ij.shape)

            har_invcov_ij = np.linalg.pinv(sliced_har_cov_ij) 
        
            del sliced_har_cov_ij

            if np.any(har_invcov_ij == 0.) or np.any(np.isnan(har_invcov_ij)):
                for l in range(len(har_invcov_ij[:,0,0])):
                    for nu_1 in range(0,len(nu_sel)):
                        for nu_2 in range(0,len(nu_sel)):
                            if har_invcov_ij[l, nu_1, nu_2] == 0.:
                                print("PS Zero for",l, nu_1, nu_2)
                            if np.isnan(har_invcov_ij[l, nu_1, nu_2]):
                                print("PS NaN for", l, nu_1, nu_2)

            # print(A_part.shape, har_invcov_ij.shape, (A_part.T).shape)
            har_wgts_part = np.matmul(A_part.T, np.copy(har_invcov_ij))
            At_Cinv_A_inv = np.linalg.inv(np.matmul(A_part.T, np.matmul(np.copy(har_invcov_ij), A_part)))

            if np.any(np.isnan(At_Cinv_A_inv)):
                for l in range(ell_min_slice-2,ell-1):
                    for nu_1 in range(0,len(nu_sel)):
                        for nu_2 in range(0,len(nu_sel)):
                            if np.isnan(At_Cinv_A_inv[l, nu_1, nu_2]):
                                print("At_Cinv_A_inv NaN for", l, nu_1, nu_2)

            har_wgts_part = np.matmul(e_vec.T, np.matmul(At_Cinv_A_inv, har_wgts_part))

            del At_Cinv_A_inv, har_invcov_ij

            # print(har_wgts_part.shape)
            har_wgts_part = np.swapaxes(np.array(har_wgts_part),0,2)[:,0,:]
            # print(har_wgts_part.shape)

            del A_part

            self.har_wgts[nu_sel, ell_min_slice:ell+1] = har_wgts_part 
        
            del har_wgts_part

    def get_cleaned_alms(self):
        """
        Combines \(a_{\\ell m}\)s for all included channels with cILC weights and 
        returns the combined \(a_{\\ell m}\).

        Returns
        -------
        complex numpy array
            A numpy 1D array of cleaned map spherical harmonic coefficients in 
            healpix C/python indexed format. 
        """
    
        for nu in range(self.nu_dim):
            if nu == 0 :
                cleaned_alm = hp.almxfl(self.alms[nu], self.har_wgts[nu])

            else :
                cleaned_alm += hp.almxfl(self.alms[nu], self.har_wgts[nu])

        return cleaned_alm

    def get_projected_alms(self, alms_to_proj, adjust_beam=True):
        """
        Combines a set \(a_{\\ell m}\)s for the included channels with cILC weights and 
        returns the combined \(a_{\\ell m}\). This method is useful to obtain residual
        of different components and noise in the cleaned maps.
        Parameters
        ----------
        alms_to_proj : complex numpy ndarray  
            Numpy 2d array of shape ``(nu_dim, len(alm))`` containing healpix spherical 
            harmonic coefficients for the component you intend to project. A total of 
            ``nu_dim`` maps for the different frequency channels.
        adjust_beam : bool, optional
            Adjust for different beam smoothing of different channels. If set to False,
            assumes that the \(a_{\\ell m}\)s all convolved to the same beam smoothing.
            Default is True.

        Returns
        -------
        complex numpy array
            A numpy 1D array of projected component spherical harmonic coefficients in 
            healpix C/python indexed format. 
        """
        ALM = hp.Alm()
        lmax_proj = ALM.getlmax(len(alms_to_proj[0,:]))
        nu_dim_proj = len(alms_to_proj[:,0])

        if lmax_proj != self.lmax:
            print("lmax of alms to project does not match lmax of weights.")
            exit()
        elif nu_dim_proj != self.nu_dim:
            print("Number of frequencies of alms to project does not match number of bands in weights.")
            exit()

    
        for nu in range(self.nu_dim):
            if nu == 0 :
                if adjust_beam:
                    cleaned_alm = hp.almxfl(alms_to_proj[nu], self.har_wgts[nu] * self.beam_ratio[nu])
                else:
                    cleaned_alm = hp.almxfl(alms_to_proj[nu], self.har_wgts[nu])

            else :
                if adjust_beam:
                    cleaned_alm += hp.almxfl(alms_to_proj[nu], self.har_wgts[nu] * self.beam_ratio[nu])
                else:
                    cleaned_alm += hp.almxfl(alms_to_proj[nu], self.har_wgts[nu])

        return cleaned_alm

# def band_map_grid(maps, label_list=None, outfile=None):
    




