#######################################################################
# This file is a part of CMBframe
#
# Cosmic Microwave Background (data analysis) frame(work)
# Copyright (C) 2021  Shamik Ghosh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information about CMBframe please visit 
# <https://github.com/1cosmologist/CMBframe> or contact Shamik Ghosh 
# at shamik@ustc.edu.cn
#
#########################################################################

import numpy as np 
import healpy as hp 

from . import cleaning_utils as cu

class gls_solver:
    '''
        Workspace for GLS solver for Linear System Inversion.
    '''

    def __init__(self, Ncov, instr=[0,1,2,3,4,5,6], comp=[0,1,2], bandpass='real'):

        lmax = len(Ncov[:,0,0]) - 1
        nu_dim = len(Ncov[0,:,0])


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

        if bandpass == 'real':
            self.A = np.copy(__A_real[instr])
        elif bandpass == 'delta':
            # print('here', __A_delta[instr], instr)
            self.A = np.copy(__A_delta[instr])
        else: 
            print("ERROR: Unsupported band type")
            exit()

        # print(self.A.shape, Ncov.shape)
        self.A = self.A[:,comp]

        # print(instr, comp, bandpass)

        Ncov_inv = np.linalg.inv(Ncov[2:])

        # print(self.A.T, Ncov_inv.shape)
        if np.any(np.isnan(Ncov_inv)):
            for l in range(0,lmax-1):
                for nu_1 in range(0,nu_dim):
                    for nu_2 in range(0,nu_dim):
                        if np.isnan(Ncov_inv[l, nu_1, nu_2]):
                            print("Noise_inv NaN for", l, nu_1, nu_2)

        self.GLS_W = np.matmul(self.A.T, Ncov_inv)

        AdNA_inv = np.linalg.inv(np.matmul(self.A.T, np.matmul(Ncov_inv, self.A)))

        if np.any(np.isnan(AdNA_inv)):
            for l in range(0,lmax-1):
                for nu_1 in range(0,nu_dim):
                    for nu_2 in range(0,nu_dim):
                        if np.isnan(AdNA_inv[l, nu_1, nu_2]):
                            print("AdNA_inv NaN for", l, nu_1, nu_2)

        self.GLS_W = np.matmul(AdNA_inv, self.GLS_W)
        self.GLS_W = np.swapaxes(np.array(self.GLS_W), 0, 1)
        self.GLS_W = np.swapaxes(self.GLS_W, 1, 2)

        zero_3x7 = np.zeros_like(self.GLS_W[:,:,0])

        self.GLS_W = np.insert(self.GLS_W, 0, zero_3x7, axis=2)
        self.GLS_W = np.insert(self.GLS_W, 0, zero_3x7, axis=2)

        # print(self.GLS_W.shape)

    def fetch_gls_weights(self, comp):
        if comp in [0,1,2]:
            return self.GLS_W[comp]
        else:
            print("ERROR: Component value must be either 0:CMB, 1:Sync or 2:Dust.")
            exit()

    def get_component(self, alms_in, comp, beams=None, com_res_beam=None):
        ALM = hp.Alm()
        lmax = ALM.getlmax(len(alms_in[0,:]))
        nu_dim = len(alms_in[:,0])

        if isinstance(beams, (tuple,np.ndarray)) and isinstance(com_res_beam, np.ndarray):
            beam_ratio = cu.beam_ratios(beams, com_res_beam)
        
        for nu in range(nu_dim):
            if nu == 0:
                proj_alms = hp.almxfl(np.copy(alms_in[nu]), self.GLS_W[comp, nu] * beam_ratio[nu])
            else:
                proj_alms += hp.almxfl(np.copy(alms_in[nu]), self.GLS_W[comp, nu] * beam_ratio[nu])

        return proj_alms




        
