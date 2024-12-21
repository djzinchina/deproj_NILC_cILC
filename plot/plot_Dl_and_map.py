import numpy as np
import healpy as hp 
import os
import cmbframe as cf
import pymaster as nmt
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm  

r = 0.0
nside = 1024
lmax = 2 * nside 
lmax_o = 1500
start_sim = 0
nsims = 300 # to avoid re-run again
show = False
map_fwhm = 11. # in arcmin

#============ Folder Parameters ==============

foreground_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/BeamSys/FG/0'
# cmb_root = '/media/doujzh/AliCPT_data/LensedCMB'
instrs = ['WMAP', 'Ali', 'HFI', 'HFI', 'Ali','HFI', 'HFI']
freqs = ['K', '95', '100', '143', '150', '217', '353']
lens_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/LensWithBeamSys'
noise_root = '/media/doujzh/Ancillary_data/cNILC_covariance_matrix/resources/Noise_sims/'
output_root = '/home/doujzh/Documents/djzfiles/plots_paper/'
data_root = '/media/doujzh/AliCPT_data/AliCPT_lens2'
mask_alicpt = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_survey_mask_C_1024.fits',field=0, dtype=np.float64)
mask_30 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_30uKcut150_C_1024.fits', field=None, dtype=np.float64)
mask_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64)
mask_UNP = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=np.float64)

# mask_c2 = nmt.mask_apodization(mask_20, 6., "C2")
# mask_UNP_c2 = nmt.mask_apodization(mask_UNP, 6., "C2")
# mask_apo = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64)

nu_dim = 7

# bins, leff, bsz, lmin = cf.setup_bins(nside, lmax_o)

def import_PSM_spec(r=0.023,file_type='unlensed'):
    # hdul = fits.open('/home/doujzh/DATA/PSM_output/components/cmb/cmb_unlensed_cl.fits')
    hdul = fits.open('/media/doujzh/AliCPT_data/data_challenge2/cl/cmb_'+file_type+'_cl.fits')
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

cl_tt, cl_ee, cl_bb_0p023, cl_te, cl_pp, cl_tp, cl_ep = import_PSM_spec()

cl_bb_lens = import_PSM_spec(file_type='lensed')[2]
cl_bb = cl_bb_0p023[0:lmax+1] * (r-0.023)/0.023 + cl_bb_lens[0:lmax+1]

ells = np.arange(lmax+1)
Dell_factor = ells * (ells + 1.) / 2. / np.pi

def fetch_cmbpfg(sim, deproj=True):
    cmbpfg = []
    for nu in range(nu_dim):
        file_name = instrs[nu]+'_'+freqs[nu]+'.fits'
        if deproj and 'Ali' in file_name: file_name = 'DEPROJ_' + file_name
        file_path = os.path.join(lens_root, str(sim).zfill(3), file_name)
        cmbpfg_nu = hp.read_map(file_path, field=None, dtype=np.float64)

        cmbpfg.append(cmbpfg_nu * mask_alicpt)

        del cmbpfg_nu

    return cmbpfg

def fetch_cmb(sim):
    file_name = 'LensInfo/CMB_LEN.fits'
    file_path = os.path.join(lens_root, str(sim).zfill(3), file_name)
    cmb = hp.read_map(file_path, field=None, dtype=np.float64)
    # cmb = cmb * mask_alicpt
    return cmb

# map_TEB = hp.read_map(data_root+'/maps/TEnilc-Bcilc_11arcmin_sim0.fits', field=(0,1,2), dtype=np.float64)
# res_TEB = hp.read_map(data_root+'/maps/tot-residual_TEnilc-Bcilc_11arcmin_sim0.fits', field=(0,1,2), dtype=np.float64)
# cmb_iqu = fetch_cmb(0)
# cmb_iqu = hp.smoothing(cmb_iqu, fwhm=np.deg2rad(map_fwhm / 60.), use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
# Bcmb = cf.get_cleanedBmap(cmb_iqu, mask_20, lmax_sht=lmax)
# TEcmb = cf.iqu2teb(cmb_iqu, mask_in=mask_30, teb='te', lmax_sht=lmax, return_alm=False)
# B_dr = hp.read_map(data_root+'/maps/deproj-residual_Bcilc_11arcmin_sim0.fits', field=None, dtype=np.float64) * mask_UNP
# B_res = hp.read_map(data_root+'/maps/tot-residual_TEnilc-Bcilc_11arcmin_sim0.fits', field=None, dtype=np.float64)[2]
# B_cilc_noise = hp.read_map(data_root+'/maps/TEnilc-Bcilc_proj-noise_11arcmin_sim0.fits', field=None, dtype=np.float64)[2]

def gridlines():
    f = plt.gcf()
    # print(f.get_children())
    cbax = f.get_children()[2] # color bar axis
    # print(cbax.get_children())
    coord_text_obj = cbax.get_children()[2] # text bar, found by the print step above
    # print(dir(coord_text_obj))
    coord_text_obj.set_fontsize(10)
    # print(coord_text_obj.get_position())
    coord_text_obj.set_position((0.5, -1.2))
    for lon in [120, 150, 180, -150, -120]:
        if 180>lon>0:
            latshift = 10
        elif lon<0:
            latshift = 0
        else:
            latshift = 5
        hp.projtext(lon+6,latshift, str(lon)+r"$^\circ$", lonlat=True)
    for lat in [30, 60]:
        if lat==30:
            lonshift = 8
        else:
            lonshift = 0
        hp.projtext(108+lonshift,lat, str(lat)+r"$^\circ$", lonlat=True)
    return
# cf.plot_maps(Bcmb, mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-0.8, vmax=0.8, resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'Bcmb_150GHz_UNP.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(TEcmb[0], mask_in=mask_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-400., vmax=400., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'Tcmb_150GHz_20cut.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(TEcmb[1], mask_in=mask_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-20., vmax=20., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'Ecmb_150GHz_20cut.png', bbox_inches='tight',pad_inches=0.)

# cf.plot_maps(map_TEB[2], mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-20., vmax=20., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'cILC-Bmap_UNP.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(map_TEB[0], mask_in=mask_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-400., vmax=400., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'NILC-Tmap_20cut.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(map_TEB[1], mask_in=mask_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-20., vmax=20., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'NILC-Emap_20cut.png', bbox_inches='tight',pad_inches=0.)

# cf.plot_maps(res_TEB[2], mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-20., vmax=20., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'cILC-Bres_UNP.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(res_TEB[0], mask_in=mask_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-400., vmax=400., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'NILC-Tres_20cut.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(res_TEB[1], mask_in=mask_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-20., vmax=20., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'NILC-Eres_20cut.png', bbox_inches='tight',pad_inches=0.)

# cf.plot_maps(B_dr, mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-0.5, vmax=0.5, resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'cILC-Bdr_UNP.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(B_cilc_noise, mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-20., vmax=20., resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'cILC-Bresnoise_UNP.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(B_res - B_cilc_noise, mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-0.5, vmax=0.5, resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'cILC-BresFGpD_UNP.png', bbox_inches='tight',pad_inches=0.)
# cf.plot_maps(B_res - B_cilc_noise - B_dr, mask_in=mask_UNP, title=None, proj='orth', unit=r'$\mu$K', vmin=-0.5, vmax=0.5, resol='print', show=False)
# gridlines()
# plt.savefig(output_root+'cILC-BresFG_UNP.png', bbox_inches='tight',pad_inches=0.)
# exit()

for sim in tqdm(range(start_sim, start_sim + nsims), ncols=120):
    if sim == 0:
            Dl_TT_mean, Dl_EE_mean, Dl_BB_mean, Dl_TE_mean, \
            resFGpD_TT_mean, resFGpD_EE_mean, resFGpD_BB_mean, resFGpD_TE_mean, \
            resNl_TT_mean, resNl_EE_mean, resNl_BB_mean = \
            [], [], [], [], \
            [], [], [], [], \
            [], [], []
    leff, Dl_TT, Dl_EE, Dl_BB, Dl_TE, \
    resFGpD_TT, resFGpD_EE, resFGpD_BB, resFGpD_TE, \
    resNl_TT, resNl_EE, resNl_BB = np.loadtxt(data_root+'/data/Dl_tot_sim'+str(sim).zfill(3)+'.dat')
    Dl_TT_mean.append(Dl_TT)
    Dl_EE_mean.append(Dl_EE)
    # Dl_BB_mean.append(Dl_BB)
    Dl_TE_mean.append(Dl_TE)
    resFGpD_TT_mean.append(resFGpD_TT)
    resFGpD_EE_mean.append(resFGpD_EE)
    # resFGpD_BB_mean.append(resFGpD_BB)
    resFGpD_TE_mean.append(resFGpD_TE)
    resNl_TT_mean.append(resNl_TT)
    resNl_EE_mean.append(resNl_EE)
    # resNl_BB_mean.append(resNl_BB)

e_Dl_TT = np.std(Dl_TT_mean, axis=0)
e_Dl_EE = np.std(Dl_EE_mean, axis=0)
e_Dl_TE = np.std(Dl_TE_mean, axis=0)
# e_Dl_BB = np.std(Dl_BB_mean, axis=0)
Dl_TT_mean = np.mean(Dl_TT_mean, axis=0)
Dl_EE_mean = np.mean(Dl_EE_mean, axis=0)
# Dl_BB_mean = np.mean(Dl_BB_mean, axis=0)
Dl_TE_mean = np.mean(Dl_TE_mean, axis=0)
resFGpD_TT_mean = np.mean(resFGpD_TT_mean, axis=0)
resFGpD_EE_mean = np.mean(resFGpD_EE_mean, axis=0)
# resFGpD_BB_mean = np.mean(resFGpD_BB_mean, axis=0)
resFGpD_TE_mean = np.mean(resFGpD_TE_mean, axis=0)
resNl_TT_mean = np.mean(resNl_TT_mean, axis=0)
resNl_EE_mean = np.mean(resNl_EE_mean, axis=0)
# resNl_BB_mean = np.mean(resNl_BB_mean, axis=0)

fig, axes = plt.subplots(2, 2, figsize=(3.5*2, 2.16*2), dpi=600, sharex='all')
fig.subplots_adjust(hspace=.07, wspace=0.3)
# print(len(axes))
plt.style.use('seaborn-v0_8-ticks')
plt.rc('font', family='sans-serif', size=6)
plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams['axes.labelsize'] = 6
# plt.rcParams['xtick.labelsize'] = 6
# plt.rcParams['ytick.labelsize'] = 6
ax = axes[0,0]
ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_tt[40:lmax_o+1], 'k-', label='TT input')
ax.errorbar(leff[1:], Dl_TT_mean[1:], yerr=np.abs(e_Dl_TT[1:]),fmt='s', lw=1., alpha=0.7, ms=2., label='NILC')
ax.plot(leff[1:], resNl_TT_mean[1:], '-', lw=1., alpha=0.7, ms=3., label='residual noise')
ax.plot(leff[1:], resFGpD_TT_mean[1:], '-', lw=1., alpha=0.7, ms=3., label='FG.+deproj. residual')
# ax.plot(leff[1:], Dl_TT_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
ax.text(0.05, 0.05, r'$TT$', ha='left', va='bottom', transform=ax.transAxes)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{TT}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='best', frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(xmin=30, xmax=lmax+10)
# ax.set_ylim(ymin=1.e0, ymax=1.e4)
# ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlTT_NILC.png',bbox_inches='tight',pad_inches=0.1)

ax = axes[0,1]
ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_ee[40:lmax_o+1], 'k-', label='EE input')
ax.errorbar(leff[1:], Dl_EE_mean[1:], yerr=np.abs(e_Dl_EE[1:]),fmt='s', lw=1., alpha=0.7, ms=2., label='NILC')
ax.plot(leff[1:], resNl_EE_mean[1:], '-', lw=1., alpha=0.7, ms=3., label='residual noise')
ax.plot(leff[1:], resFGpD_EE_mean[1:], '-', lw=1., alpha=0.7, ms=3., label='FG.+deproj. residual')
# ax.plot(leff[1:], Dl_EE_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
ax.text(0.05, 0.05, r'$EE$', ha='left', va='bottom', transform=ax.transAxes)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{EE}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='upper left', frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(xmin=30, xmax=lmax+10)
# ax.set_ylim(ymin=1.e-3, ymax=1.e2)
# ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlEE_NILC.png',bbox_inches='tight',pad_inches=0.1)

ax = axes[1,0]
ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*(cl_te)[40:lmax_o+1], 'k-', label='TE input')
ax.errorbar(leff[1:], Dl_TE_mean[1:], yerr=np.abs(e_Dl_TE[1:]),fmt='s', lw=1., alpha=0.7, ms=2., label='NILC')
ax.plot(leff[1:], (resFGpD_TE_mean)[1:], '-',color='C2', lw=1., alpha=0.7, ms=3., label='FG.+deproj. residual')
# ax.plot(leff[1:], Dl_TE_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
ax.text(0.05, 0.05, r'$TE$', ha='left', va='bottom', transform=ax.transAxes)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{TE}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='best', frameon=False)
ax.set_xscale('log')
ax.set_xlim(xmin=30, xmax=lmax+10)
# ax.set_yscale('log')
# ax.set_ylim(ymin=5.e-2, ymax=2.e2)
# ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlTE_NILC.png',bbox_inches='tight',pad_inches=0.1)

for sim in tqdm(range(start_sim, start_sim + nsims), ncols=120):
    if sim == 0:
            Dl_BB_mean, resFGpD_BB_mean, resNl_BB_mean = [], [], []
            Dl_BB_dr_mean, resFG_BB_mean = [], []
    Dl_BB = np.loadtxt(data_root+'/data/no_fix_bin/Dl_tot_sim'+str(sim).zfill(3)+'.dat')[3]
    Dl_BB_mean.append(Dl_BB)
    leff, Dl_BB_dr, resNl_BB, resFGpD_BB, resFG_BB = np.loadtxt(data_root+'/data/no_fix_bin/Dl_BB_deproj_res_sim'+str(sim).zfill(3)+'.dat')
    resNl_BB_mean.append(resNl_BB)
    Dl_BB_dr_mean.append(Dl_BB_dr)
    resFGpD_BB_mean.append(resFGpD_BB)
    resFG_BB_mean.append(resFG_BB)

e_Dl_BB = np.std(Dl_BB_mean, axis=0)
Dl_BB_mean = np.mean(Dl_BB_mean, axis=0)
resFGpD_BB_mean = np.mean(resFGpD_BB_mean, axis=0)
resFG_BB_mean = np.mean(resFG_BB_mean, axis=0)
resNl_BB_mean = np.mean(resNl_BB_mean, axis=0)
Dl_BB_dr_mean = np.mean(Dl_BB_dr_mean, axis=0)

ax = axes[1,1]
ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_bb[40:lmax_o+1], 'k-', label='BB input')
ax.errorbar(leff[1:], Dl_BB_mean[1:], yerr=np.abs(e_Dl_BB[1:]),fmt='s', lw=1., alpha=0.7, ms=2., label='cILC')
ax.plot(leff[1:], resNl_BB_mean[1:], '-', lw=1., alpha=0.7, ms=3., label='residual noise')
ax.plot(leff[1:], resFGpD_BB[1:], '-', lw=1., alpha=0.7, ms=3., label='FG.+deproj. residual')
ax.plot(leff[1:], Dl_BB_dr_mean[1:], '-', lw=1., alpha=0.7, ms=3., label='deproj. residual')
ax.plot(leff[1:], resFG_BB[1:], '-', lw=1., alpha=0.7, ms=3., label='FG. residual')
# ax.plot(leff[1:], Dl_BB_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
ax.text(0.05, 0.05, r'$BB$', ha='left', va='bottom', transform=ax.transAxes)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='best', frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(xmin=30, xmax=lmax+10)
# ax.set_ylim(ymin=1.e-3, ymax=1.e2)
# ax.grid(which='both', axis='both')
plt.savefig(output_root+'/Dl_all.png',bbox_inches='tight',pad_inches=0.1)