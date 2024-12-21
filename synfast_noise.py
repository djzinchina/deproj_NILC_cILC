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

# overwrite_inp = True
constrain = [1,2]

cmb_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/CMB/'
lens_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/LensWithBeamSys/'
output_root = '/home/doujzh/Documents/AliCPT_beamsys/output/'
noise_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/Noise/'
mask_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/Mask/'
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

msk = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=None, verbose=False)
msk_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64, verbose=False)
msk_20c2 = nmt.mask_apodization(msk_20, 6.0, apotype='C2')
msk_apo = hp.read_map('/home/doujzh/DATA/AliCPT/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64, verbose=False)

foreground_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/FG/0/'
# cmb_root = '/media/doujzh/AliCPT_data/LensedCMB'
instrs = ['WMAP', 'Ali', 'HFI', 'HFI', 'Ali','HFI', 'HFI']
freqs = ['K', '95', '100', '143', '150', '217', '353']
map_sel = np.arange(len(freqs))
bands_beam = [52.8, 19., 9.682, 7.303, 11., 5.021, 4.944]

lmax_ch = np.array([350, 1200, lmax, lmax, lmax, lmax, lmax])
# lmax_ch = np.array([lmax, lmax, lmax, lmax, lmax, lmax, lmax])
nu_dim = len(map_sel)
fg = []
beams = []
for nu in map_sel:
    # file_name = instrs[nu]+'_'+freqs[nu]+'.fits'
    # file_path = os.path.join(foreground_root, file_name)

    # fg.append(hp.read_map(file_path, field=None, dtype=np.float64))

    beams.append(hp.gauss_beam(np.deg2rad(bands_beam[nu] / 60.), pol=True, lmax=lmax))

# exit()
beams = np.array(beams)
beam_0 = hp.gauss_beam(np.deg2rad(map_fwhm / 60.), pol=True, lmax=lmax)
def fetch_cmb(sim):
    cmb = []
    for nu in map_sel:
        file_name = instrs[nu]+'_'+freqs[nu]+'.fits'
        file_path = os.path.join(lens_root, str(sim).zfill(3), file_name)
        cmb_nu = hp.read_map(file_path, field=None, dtype=np.float64)

        cmb.append(cmb_nu * msk_alicpt)

        del cmb_nu

    return cmb
def fetch_noise(sim):
    noise = []
    for nu in map_sel:
        file_name = instrs[nu]+'_'+freqs[nu]+'.fits'
        file_path = os.path.join(noise_root, str(sim), file_name)
        noise_nu = hp.read_map(file_path, field=None, dtype=np.float64)

        noise.append(noise_nu * msk_alicpt)

        del noise_nu

    return noise
##################### COMPUTING THE COVARIANCE MATRIX ##########################

# print(nu_dim)

# print(nu_dim)
bins, leff, bsz, lmins = cf.setup_bins(nside, lmax_o, loglims=[[1.+bin_size/100]], bsz_0=bin_size)
ell = np.arange(lmax_o+1)
Dell_factor = ell * (ell + 1.) / 2. / np.pi
DlBB_lensed = cl_bb_lensed[:lmax_o+1] * Dell_factor
DlBB_lens = cf.binner(DlBB_lensed, lmax_o, bsz, leff, is_Cell = False)

def compute_Dell(Bmap, beam=11.):
    fld_B = nmt.NmtField(msk_apo, [Bmap], lmax_sht=lmax, masked_on_input=False,)
    B_wsp = nmt.NmtWorkspace()
    B_wsp.compute_coupling_matrix(fld_B, fld_B, bins)

    Cl_coup_BB = nmt.compute_coupled_cell(fld_B, fld_B)/hp.gauss_beam(np.deg2rad(beam / 60.), pol=True, lmax=lmax)[:,2]**2
    Dl_BB_nmt = B_wsp.decouple_cell(Cl_coup_BB)[0]
    return Dl_BB_nmt
# Dl_BB_act = B_wsp.decouple_cell(Cl_coup_BB, cl_noise=Nl_coup_BB)[0]
sigma_n = hp.read_map("/media/doujzh/AliCPT_data2/Zirui_beamsys/Noise/SIGMA_Ali_150.fits", field=None, dtype=np.float64)
npix = hp.nside2npix(nside)
mask = np.zeros(npix)
mask[sigma_n[1]<30.] = 1.
mask[sigma_n[1]<=0.] = 0.
hp.write_map(mask_root+"AliCPT_30uKP150cut_C_1024.fits", mask, overwrite=True, dtype=np.float64)
cf.plot_maps(mask, proj='moll', outfile=mask_root+"AliCPT_30uKP150cut_C_1024.png", show=False)
# sigma_n[np.isinf(sigma_n)] = 0.
# hp.write_map("/media/doujzh/AliCPT_data2/Zirui_beamsys/Noise/SIGMA_Ali_95.fits", sigma_n, overwrite=True, dtype=np.float64)
# sigma_n = hp.read_map("/media/doujzh/AliCPT_data/NoiseVarDC1/I_NOISE_150_C_1024.fits", field=None, dtype=np.float64)
# sigma_n = hp.read_map("/media/doujzh/AliCPT_data/AliCPT_widescan/20211030/WideScan2/AliCPT_1_150GHz_NOISE.fits", field=None, dtype=np.float64)
# sigma_n2 = hp.read_map("/media/doujzh/AliCPT_data/FullScan/NoiseLevelMap20230309/AliCPT_1_150GHz_HPX1024_FULL_1MODYR_NSTD_P.fits", field=None, dtype=np.float64)
# print(sigma_n.shape)
# sigma_n[:,msk_wide==0.] = 0.
# cf.plot_maps(sigma_n[0], proj='moll', vmax=20, outfile=output_root+'/maps/SigmaI_new_150GHz.png', show=False)
# cf.plot_maps(sigma_n[0], proj='moll', vmax=20, outfile=output_root+'/maps/SigmaI_wide1_150GHz.png', show=False)
exit()
# print(sigma_n[0]**2 / (sigma_n[1]**2+sigma_n[2]**2))

def get_sigma(channel, pol=True):
    datafile = noise_root+"SIGMA_"+channel+".fits"
    if pol:
        sigma_I, sigma_Q, sigma_U = hp.read_map(datafile, field=None, dtype=np.float64)
        return np.array([sigma_I, sigma_Q, sigma_U])
    else:
        sigma_I = hp.read_map(datafile, field=0, dtype=np.float64)
        return np.array(sigma_I)

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

def get_noise(channel, nside, pol=True, seed=None):
    nside_ali = 1024

    if nside < nside_ali:
        nside_o = nside
        nside = nside_ali
        need_to_downgrade = True 
    else:
        need_to_downgrade = False

    npix = hp.nside2npix(nside)

    sigma = get_sigma(channel, pol=pol)
    # if 'Ali' in channel: sigma[np.isinf(sigma)] = 0.
    if channel == 'WMAP_K': sigma = upscale_sigma(sigma, nside_ali, nside_in=512)

    if nside > nside_ali:
        sigma = upscale_sigma(sigma, nside)

    if seed is None:
        rng = default_rng()
    else:
        rng = default_rng(seed)
        
    if not pol:
        noise_map = rng.standard_normal(size=(npix,), dtype=np.float64) * sigma
    else:
        noise_map = rng.standard_normal(size=(3, npix), dtype=np.float64) * sigma

    if need_to_downgrade:
        noise_map = hp.ud_grade(noise_map, nside_o)

    return np.array(noise_map)

# generate noise
for nu in map_sel:
    channel = instrs[nu] + '_' + freqs[nu]
    for i in tqdm(range(100)):
        noise_nu = get_noise(channel, nside, pol=True)
        noise_path = noise_root+str(i)+'/'
        Path(noise_path).mkdir(parents=True, exist_ok=True)
        noise_filename = noise_path + channel + '.fits'
        if i == 0 and nu == 4:
            cf.plot_maps(noise_nu[0], proj='moll', vmin=-20, vmax=20, outfile=output_root+'/maps/NoiseI_150GHz.png', show=False)
            cf.plot_maps(noise_nu[1], proj='moll', vmin=-20, vmax=20, outfile=output_root+'/maps/NoiseQ_150GHz.png', show=False)
            cf.plot_maps(noise_nu[2], proj='moll', vmin=-20, vmax=20, outfile=output_root+'/maps/NoiseU_150GHz.png', show=False)
        hp.write_map(noise_filename, noise_nu, overwrite=True, dtype=np.float64)