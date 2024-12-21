import numpy as np 
import healpy as hp
import cmbframe as cf 
import pymaster as nmt
from tqdm import tqdm
import astropy.io.fits as fits 
import matplotlib.pyplot as plt
import chilc
import subprocess as sp
import os.path as op
import os 
from pathlib import Path
from numpy.random import default_rng

import warnings
warnings.simplefilter("ignore")

nside = 1024
lmax_rot = 2*nside
lmax = 1000
lmax_o = 700
nsims = 100
start_sim = 0
r=0.0
bin_size = 30
map_fwhm = 52.8 # arcmin
npix = hp.nside2npix(nside)
# overwrite_inp = True
constrain = [1,2]

cmb_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/CMB/'
lens_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/LensWithBeamSys/'
output_root = '/home/doujzh/Documents/AliCPT_beamsys/output/'
noise_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/Noise/'
mask_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/Mask/'
foreground_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/FG/0/'
def import_PSM_spec(r=0.023,file_type='unlensed'):
    # hdul = fits.open('/home/doujzh/DATA/PSM_output/components/cmb/cmb_unlensed_cl.fits')
    hdul = fits.open('/media/doujzh/AliCPT_data2/data_challenge2/cl/cmb_'+file_type+'_cl.fits')
    # cols = hdul[1].columns
    # cols.info()
    data = hdul[1].data
    cl_tt = np.array(data['C_1_1'])[0]
    cl_ee = np.array(data['C_2_2'])[0]
    cl_bb = np.array(data['C_3_3'])[0] * (r / 0.023)
    cl_te = np.array(data['C_1_2'])[0]
    cl_pp = np.array(data['C_4_4'])[0]
    cl_tp = np.array(data['C_1_4'])[0]
    cl_ep = np.array(data['C_2_4'])[0]
    return cl_tt, cl_ee, cl_bb, cl_te, cl_pp, cl_tp, cl_ep

cl_tt, cl_ee, cl_bb, cl_te, cl_pp, cl_tp, cl_ep = import_PSM_spec()

cl_bb_lens = import_PSM_spec(file_type='lensed')[2]
cl_bb_lensed = cl_bb[0:lmax_rot+1] * (r-0.023)/0.023 + cl_bb_lens[0:lmax_rot+1]
msk_alicpt = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_survey_mask_C_1024.fits',field=0, dtype=np.float64, verbose=False)

msk_unp = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=None, verbose=False)
msk_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64, verbose=False)
msk_20c2 = nmt.mask_apodization(msk_20, 6.0, apotype='C2')
msk_unpinv = hp.read_map('/home/doujzh/DATA/AliCPT/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64, verbose=False)
msk_old = hp.read_map('/home/doujzh/DATA/AliCPT/NoiseVar_MERRA_2/40Hz/AliCPT_survey_mask_C_1024.fits',field=0, dtype=None, verbose=False)
cf.plot_maps(msk_old, proj='moll', outfile=mask_root+"AliCPT_Mask_DC1.png", show=False)
cf.plot_maps(msk_20, proj='moll', outfile=mask_root+"AliCPT_20uKMask_DC1.png", show=False)
cf.plot_maps(msk_unp, proj='moll', outfile=mask_root+"AliCPT_UNPMask_DC1.png", show=False)
exit()
# cmb_root = '/media/doujzh/AliCPT_data/LensedCMB'
instrs = ['WMAP', 'Ali', 'HFI', 'HFI', 'Ali','HFI', 'HFI']
freqs = ['K', '95', '100', '143', '150', '217', '353']
map_sel = np.arange(len(freqs))
bands_beam = [52.8, 19., 9.682, 7.303, 11., 5.021, 4.944]

sigma_n = hp.read_map("/media/doujzh/AliCPT_data2/Zirui_beamsys/Noise/SIGMA_Ali_150.fits", field=None, dtype=np.float64)
# npix = hp.nside2npix(nside)
# mask = np.zeros(npix)
# mask[sigma_n[1]<30.] = 1.
# mask[sigma_n[1]<=0.] = 0.
# hp.write_map(mask_root+"AliCPT_30uKP150cut_C_1024.fits", mask, overwrite=True, dtype=np.float64)
# cf.plot_maps(mask, proj='moll', outfile=mask_root+"AliCPT_30uKP150cut_C_1024.png", show=False)

# fg = hp.read_map(foreground_root+"Ali_150.fits", field=None, dtype=np.float64)
# P_fg = np.sqrt(fg[1]**2 + fg[2]**2)
# P_fg *= cf.convert_KCMB_to_MJysr('AliCPT', '150') * cf.convert_MJysr_to_Kb('AliCPT', '150') * 1.e-3 # from uKCMB to mKRJ
# cf.plot_maps(P_fg, title='AliCPT beamsys', proj='moll', vmax=0.01, unit='mK_RJ', outfile=mask_root+"P-fgs_AliCPT_150GHz_C_1024.png", show=False)

rot = hp.Rotator(coord=['G','C'])
################ SIMCA FGs ##############
# unit: mK_RJ Nside: 2048 Cor: Galactic FWHM: 40arcmin Freq: 30GHz
# smica_sync_Q, smica_sync_U = hp.read_map(mask_root+"COM_CompMap_QU-synchrotron-smica_2048_R3.00_full.fits", field=None, dtype=np.float64, verbose=False)
# smica_sync_P = np.sqrt(smica_sync_Q**2. + smica_sync_U**2.)
# smica_sync_P = hp.ud_grade(smica_sync_P, nside)
# Psync_C = rot.rotate_map_alms(smica_sync_P, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts', lmax=lmax)
# hp.write_map(mask_root+"SMICA2018_P-synchrotron_C_1024.fits", Psync_C, overwrite=True, dtype=np.float64)
# Psync_C = hp.read_map(mask_root+"SMICA2018_P-synchrotron_C_1024.fits", field=0, dtype=np.float64, verbose=False)
# Psync_C = cf.harmonic_udgrade(Psync_C, fwhm_in=40., fwhm_out=5*60.)
# cf.plot_maps(Psync_C, title='Sync Smooth 5deg', proj='moll', vmax=0.03, unit='mK_RJ', outfile=mask_root+"SMICA2018_P-synchrotron_C_1024.png", show=False)

# unit: mK_RJ Nside: 2048 Cor: Galactic FWHM: 12arcmin Freq: 353GHz
# smica_dust_Q, smica_dust_U = hp.read_map(mask_root+"COM_CompMap_QU-thermaldust-smica_2048_R3.00_full.fits", field=None, dtype=np.float64, verbose=False)
# smica_dust_P = np.sqrt(smica_dust_Q**2. + smica_dust_U**2.)
# smica_dust_P = hp.ud_grade(smica_dust_P, nside)
# Pdust_C = rot.rotate_map_alms(smica_dust_P, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts', lmax=lmax)
# hp.write_map(mask_root+"SMICA2018_P-thermaldust_C_1024.fits", Pdust_C, overwrite=True, dtype=np.float64)
# Pdust_C = hp.read_map(mask_root+"SMICA2018_P-thermaldust_C_1024.fits", field=0, dtype=np.float64, verbose=False)
# Pdust_C = cf.harmonic_udgrade(Pdust_C, fwhm_in=12., fwhm_out=5*60.)
# cf.plot_maps(Pdust_C, title='Dust Smooth 5deg', proj='moll', vmax=0.03, unit='mK_RJ', outfile=mask_root+"SMICA2018_P-thermaldust_C_1024.png", show=False)

############# GNILC PR3 Dust ################
# IQU..., unit = KCMB, FWHM = 80 arcmin, Freq = 353GHz, No color correction
# gnilc_dust_Q, gnilc_dust_U = hp.read_map(mask_root+"COM_CompMap_IQU-thermaldust-gnilc-unires_2048_R3.00.fits", field=None, dtype=np.float64, verbose=False)[1:3]
# gnilc_dust_P = np.sqrt(gnilc_dust_Q**2. + gnilc_dust_U**2.)
# gnilc_dust_P = hp.ud_grade(gnilc_dust_P, nside)
# Pdust_C = rot.rotate_map_alms(gnilc_dust_P, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts', lmax=lmax)
# Pdust_C *= cf.convert_KCMB_to_MJysr('HFI', '353') * cf.convert_MJysr_to_Kb('HFI', '353') * 1.e3 # from KCMB to mKRJ
# hp.write_map(mask_root+"GNILC_P-thermaldust_353GHz_C_1024.fits", Pdust_C, overwrite=True, dtype=np.float64)
# Pdust_C = cf.harmonic_udgrade(Pdust_C, fwhm_in=80., fwhm_out=5*60.)
# cf.plot_maps(Pdust_C, title='Dust Smooth 5deg', proj='moll', vmax=0.03, unit='mK_RJ', outfile=mask_root+"GNILC_P-thermaldust_353GHz_C_1024.png", show=False)
######## scale with MBB SED, T = 19.6K and beta_P = 1.53 from Planck 2018 XI #################
Pdust_C = hp.read_map(mask_root+"GNILC_P-thermaldust_353GHz_C_1024.fits", field=0, dtype=np.float64, verbose=False)
def mbb_scale(nu): # in Kb
    beta_d = 1.53
    T_d = 19.6
    return cf.modified_blackbody(nu, beta_d, T_d)*cf.convert_MJysr_to_Kb('HFI', str(nu))
Pdust_C *= mbb_scale(143)/mbb_scale(353)
Pdust_C = cf.harmonic_udgrade(Pdust_C, fwhm_in=80., fwhm_out=5*60.)
cf.plot_maps(Pdust_C, title='Dust Smooth 5deg', proj='moll', vmax=0.01, unit='mK_RJ', outfile=mask_root+"GNILC_P-thermaldust_143GHz_C_1024.png", show=False)

############### Cosmoglobe DR1 Sync #####################
# I/Q/U/P/I_beta/QU_beta/..., unit = uKRJ, FWHM = 60 arcmin, Freq_P = 30GHz
# hdul = fits.open(mask_root+"CG_synch_IQU_n1024_v1.fits")
# print(hdul[0].header)
# print(hdul[1].header)
# cg_sync_P, cg_sync_I_beta, cg_sync_P_beta = hp.read_map(mask_root+"CG_synch_IQU_n1024_v1.fits", field=None, dtype=np.float64, verbose=False)[3:6]
# Psync_C = rot.rotate_map_alms(cg_sync_P, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts', lmax=lmax)
# Psync_C *= 1.e-3
# hp.write_map(mask_root+"CG_P-synchrotron_30GHz_C_1024.fits", Psync_C, overwrite=True, dtype=np.float64)
# Psync_C = cf.harmonic_udgrade(Psync_C, fwhm_in=60., fwhm_out=5*60.)
# cf.plot_maps(Psync_C, title='Sync Smooth 5deg', proj='moll', vmax=0.03, unit='mK_RJ', outfile=mask_root+"CG_P-synchrotron_30GHz_C_1024.png", show=False)
########## scale with PL SED, beta = -3.07 from Cosmoglobe DR1 III ###################
Psync_C = hp.read_map(mask_root+"CG_P-synchrotron_30GHz_C_1024.fits", field=0, dtype=np.float64, verbose=False)
def pl_scale(nu): # in Kb
    beta_s = -3.07
    return cf.powerlaw(nu, 353., spec_ind=beta_s)
Psync_C *= pl_scale(143.)/pl_scale(30.)
Psync_C = cf.harmonic_udgrade(Psync_C, fwhm_in=60., fwhm_out=5*60.)
cf.plot_maps(Psync_C, title='Sync Smooth 5deg', proj='moll', vmax=0.01, unit='mK_RJ', outfile=mask_root+"CG_P-synchrotron_143GHz_C_1024.png", show=False)
############# SUM of 2 FGs ################
hp.write_map(mask_root+"P-fgs_5deg_143GHz_C_1024.fits", Pdust_C + Psync_C, overwrite=True, dtype=np.float64)
cf.plot_maps(Pdust_C + Psync_C, title='FG Smooth 5deg', proj='moll', vmax=0.01, unit='mK_RJ', outfile=mask_root+"P-fgs_5deg_143GHz_C_1024.png", show=False)

############ Produce cutfg mask #######################
Pfg_100 = hp.read_map(mask_root+"P-fgs_5deg_100GHz_C_1024.fits", field=0, dtype=np.float64, verbose=False)
Pfg_143 = hp.read_map(mask_root+"P-fgs_5deg_143GHz_C_1024.fits", field=0, dtype=np.float64, verbose=False)

mask = hp.read_map(mask_root+"AliCPT_10uKP150cut_C_1024.fits", field=0, dtype=np.float64, verbose=False)


hole = np.ones(npix)
hole[Pfg_100>0.01] = 0.
hole[Pfg_143>0.01] = 0.
cf.plot_maps(hole, proj='moll', outfile=mask_root+"Cut_part.png", show=False)
hole = nmt.mask_apodization(hole, 2.0, apotype='C2')
hole[hole<1.] = 0.
cf.plot_maps(hole, proj='moll', outfile=mask_root+"Cutapo_part.png", show=False)

# hole[hole>=0.9] = 1.
mask = hole*mask
# mask = nmt.mask_apodization(mask, 4.0, apotype='C2')
hp.write_map(mask_root+"Mask_test.fits", mask, overwrite=True, dtype=np.float64)
cf.plot_maps(mask, proj='moll', outfile=mask_root+"Mask_test.png", show=False)
mask /= sigma_n[1]
print(mask)
mask[np.isnan(mask)] = 0.
hp.write_map(mask_root+"Maskapo_test.fits", mask, overwrite=True, dtype=np.float64)
cf.plot_maps(mask, proj='moll', outfile=mask_root+"Maskapo_test.png", show=False)


def upscale_sigma(sigma, nside_out, nside_in=1024):
    npix = hp.nside2npix(nside_out)

    pix_arr = np.arange(npix)
    x,y,z = hp.pix2vec(nside_out, pix_arr)
    pix_map = hp.vec2pix(nside_in, x, y, z)

    pixratio = nside_out / nside_in

    if sigma.ndim == 1:
        upscaled_sigma = sigma[pix_map] * pixratio
    else:
        upscaled_sigma = sigma[:, pix_map] * pixratio

    return upscaled_sigma

