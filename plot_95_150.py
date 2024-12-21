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
freq = '150'
fwhm = {'95':19., '150':11.}[freq]
sim = 0

cmb_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/CMB/'
lens_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/LensWithBeamSys/'
output_root = '/home/doujzh/Documents/AliCPT_beamsys/output/'

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

cl_tt, cl_ee, cl_bb_lens, cl_te, cl_pp, cl_tp, cl_ep = import_PSM_spec(file_type='lensed')
cl_bb_lensed = cl_bb[0:lmax_rot+1] * (r-0.023)/0.023 + cl_bb_lens[0:lmax_rot+1]

msk_alicpt = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_survey_mask_C_1024.fits',field=0, dtype=np.float64, verbose=False)
msk3 = [msk_alicpt, msk_alicpt, msk_alicpt]

msk = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=None, verbose=False)
msk_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64, verbose=False)
msk_20c2 = nmt.mask_apodization(msk_20, 6.0, apotype='C2')
msk_apo = hp.read_map('/home/doujzh/DATA/AliCPT/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64, verbose=False)

msk_arr = [msk_apo, msk_apo, msk_apo]

foreground_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/FG/0/'
# cmb_root = '/media/doujzh/AliCPT_data/LensedCMB'

# print(nu_dim)

# print(nu_dim)
# bins, leff, bsz, lmins = cf.setup_bins(nside, lmax_o, loglims=[[1.+bin_size/100]], bsz_0=bin_size)
# ell = np.arange(lmax_o+1)
# Dell_factor = ell * (ell + 1.) / 2. / np.pi
# DlBB_lensed = cl_bb_lensed[:lmax_o+1] * Dell_factor
# DlBB_lens = cf.binner(DlBB_lensed, lmax_o, bsz, leff, is_Cell = False)
# DlTT = cl_tt[:lmax_o+1] * Dell_factor
# DlTT_lens = cf.binner(DlTT, lmax_o, bsz, leff, is_Cell = False)
# DlEE = cl_ee[:lmax_o+1] * Dell_factor
# DlEE_lens = cf.binner(DlEE, lmax_o, bsz, leff, is_Cell = False)
bins, leff = cf.setup_bins(nside, lmax_o=lmax_o, bsz_0=30, fixed_bins=True)
# bins, leff, bsz, lmins = cf.setup_bins(nside, lmax_o, loglims=[[1.+bin_size/100]], bsz_0=bin_size)
ells = np.arange(lmax+1)
Dell_factor = ells * (ells + 1.) / 2. / np.pi
def Dl_binner(cl_input, is_Cell=True):
    if is_Cell:
        Dl = cl_input[:lmax+1] * Dell_factor
    else:
        Dl = cl_input[:lmax+1]
    Dl_bin = cf.binner(Dl, lmax_o, 30, leff, is_Cell = False, fixed_bins=True)
    return Dl, Dl_bin
DlTT, DlTT_bin = Dl_binner(cl_tt)
DlEE, DlEE_bin = Dl_binner(cl_ee)
DlBB, DlBB_bin = Dl_binner(cl_bb_lensed)

def compute_Dell(Bmap, beam=11., TEB=2):
    fld_B = nmt.NmtField(msk_apo, [Bmap], lmax_sht=lmax, masked_on_input=False,)
    B_wsp = nmt.NmtWorkspace()
    B_wsp.compute_coupling_matrix(fld_B, fld_B, bins)

    Cl_coup_BB = nmt.compute_coupled_cell(fld_B, fld_B)/hp.gauss_beam(np.deg2rad(beam / 60.), pol=True, lmax=lmax)[:,TEB]**2
    Dl_BB_nmt = B_wsp.decouple_cell(Cl_coup_BB)[0]
    return Dl_BB_nmt
# Dl_BB_act = B_wsp.decouple_cell(Cl_coup_BB, cl_noise=Nl_coup_BB)[0]
# sigma_n = hp.read_map("/media/doujzh/AliCPT_data2/Zirui_beamsys/Noise/SIGMA_HFI_100.fits", field=None, dtype=np.float64)
# print(sigma_n.shape)
# print(sigma_n[0]**2 / (sigma_n[1]**2+sigma_n[2]**2))

IQU = hp.read_map(lens_root+str(sim).zfill(3)+'/Ali_'+freq+'.fits', field=None, dtype=np.float64)
Bmap_bs = cf.get_cleanedBmap(IQU, msk_20, lmax_sht=lmax_rot)
cf.plot_maps(Bmap_bs*msk_20c2, mask_in=msk_20, proj='orth',  outfile=output_root+'/maps/Bcmbpfg_beamsys_'+freq+'GHz.png', show=False)
DlBB_bs = compute_Dell(Bmap_bs, fwhm, 2)
TEmap = cf.iqu2teb(IQU, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
DlTT_bs = compute_Dell(TEmap[0], fwhm, 0)
DlEE_bs = compute_Dell(TEmap[1], fwhm, 1)

IQU = hp.read_map(lens_root+str(sim).zfill(3)+'/DEPROJ_Ali_'+freq+'.fits', field=None, dtype=np.float64)
Bmap_dp = cf.get_cleanedBmap(IQU, msk_20, lmax_sht=lmax_rot)
cf.plot_maps(Bmap_dp*msk_20c2, mask_in=msk_20, proj='orth',  outfile=output_root+'/maps/Bcmbpfg_deproj_'+freq+'GHz.png', show=False)
DlBB_dp = compute_Dell(Bmap_dp, fwhm, 2)
TEmap = cf.iqu2teb(IQU, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
DlTT_dp = compute_Dell(TEmap[0], fwhm, 0)
DlEE_dp = compute_Dell(TEmap[1], fwhm, 1)

IQU = hp.read_map(foreground_root+'Ali_'+freq+'.fits', field=None, dtype=np.float64)
Bmap_fg = cf.get_cleanedBmap(IQU, msk_20, lmax_sht=lmax_rot)
cf.plot_maps(Bmap_fg*msk_20c2, mask_in=msk_20, proj='orth',  outfile=output_root+'/maps/Bfg_beamsys_'+freq+'GHz.png', show=False)
DlBB_fg = compute_Dell(Bmap_fg, fwhm, 2)
TEmap = cf.iqu2teb(IQU, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
DlTT_fg = compute_Dell(TEmap[0], fwhm, 0)
DlEE_fg = compute_Dell(TEmap[1], fwhm, 1)
del Bmap_fg, Bmap_dp, Bmap_bs, IQU, TEmap

fig, axes = plt.subplots(3, 1, figsize=(3.5, 2.1*3), dpi=600, sharex='all')
fig.subplots_adjust(hspace=.07)
ax = axes[2]
# ax.plot(ell[30:], DlBB_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
ax.plot(leff[1:], (DlBB_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
ax.plot(leff[1:], (DlBB_bin+DlBB_fg)[1:], 'k-', lw=1., alpha=0.7, label='theo.CMB + beamsys.FG. '+freq+'GHz')
ax.plot(leff[1:], DlBB_bs[1:], '--', lw=1., alpha=0.7, label='Add Beamsys '+freq+'GHz')
ax.plot(leff[1:], DlBB_dp[1:], '--', lw=1., alpha=0.7, label='Deproj. '+freq+'GHz')
ax.plot(leff[1:], (DlBB_fg)[1:], '--', lw=1., alpha=0.7, label='beamsys.FG. '+freq+'GHz')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
# ax.legend(loc='best', frameon=False, fontsize=7)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(ymin=1.e-7, ymax=1.e2)
# ax.grid(which='both', axis='both')

ax = axes[1]
# ax.plot(ell[30:], DlEE_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
ax.plot(leff[1:], (DlEE_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
ax.plot(leff[1:], (DlEE_bin+DlEE_fg)[1:], 'k-', lw=1., alpha=0.7, label='theo.CMB + beamsys.FG. '+freq+'GHz')
ax.plot(leff[1:], DlEE_bs[1:], '--', lw=1., alpha=0.7, label='Add Beamsys '+freq+'GHz')
ax.plot(leff[1:], DlEE_dp[1:], '--', lw=1., alpha=0.7, label='Deproj. '+freq+'GHz')
ax.plot(leff[1:], (DlEE_fg)[1:], '--', lw=1., alpha=0.7, label='beamsys.FG. '+freq+'GHz')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{EE}_\ell$ [in $\mu$K${}^2$]')
# ax.legend(loc='best', frameon=False, fontsize=7)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(ymin=1.e-7, ymax=1.e2)
# ax.grid(which='both', axis='both')

ax = axes[0]
# ax.plot(ell[30:], DlTT_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
ax.plot(leff[1:], (DlTT_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
ax.plot(leff[1:], (DlTT_bin+DlTT_fg)[1:], 'k-', lw=1., alpha=0.7, label='theo.CMB + beamsys.FG. '+freq+'GHz')
ax.plot(leff[1:], DlTT_bs[1:], '--', lw=1., alpha=0.7, label='Add Beamsys '+freq+'GHz')
ax.plot(leff[1:], DlTT_dp[1:], '--', lw=1., alpha=0.7, label='Deproj. '+freq+'GHz')
ax.plot(leff[1:], (DlTT_fg)[1:], '--', lw=1., alpha=0.7, label='beamsys.FG. '+freq+'GHz')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{TT}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='best', frameon=False, fontsize=7)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(ymin=1.e-7, ymax=1.e2)
# ax.grid(which='both', axis='both')
plt.savefig(output_root+'Dl_'+freq+'GHz_bsz'+str(bin_size)+'.png',bbox_inches='tight',pad_inches=0.1)

# plt.show()







