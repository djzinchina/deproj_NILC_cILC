import numpy as np 
import healpy as hp
import cmbframe as cf 
import pymaster as nmt
import astropy.io.fits as fits 
import matplotlib.pyplot as plt
import os
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
freqs = ['95','150']
# freqs = ['150']
fwhms = {'95':19., '150':11.}
sim = 0

# cmb_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/BeamSys/CMB/'+str(sim)+'/' # r=0.023 wo lensing
foreground_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/BeamSys/FG/0/'
fg_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/WOBeamSys/FG/'
lens_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/LensWithBeamSys/'
output_root = '/home/doujzh/Documents/djzfiles/plots_paper/'

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

cl_tt, cl_ee, cl_bb, cl_te, cl_pp, cl_tp, cl_ep = import_PSM_spec()

cl_tt, cl_ee, cl_bb_lens, cl_te, cl_pp, cl_tp, cl_ep = import_PSM_spec(file_type='lensed')
cl_bb_lensed = cl_bb[0:lmax_rot+1] * (r-0.023)/0.023 + cl_bb_lens[0:lmax_rot+1]

msk_alicpt = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_survey_mask_C_1024.fits',field=0, dtype=np.float64, verbose=False)
msk3 = [msk_alicpt, msk_alicpt, msk_alicpt]

msk = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=None, verbose=False)
msk_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64, verbose=False)
msk_20c2 = nmt.mask_apodization(msk_20, 6.0, apotype='C2')
msk_inv = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64, verbose=False)
fsky = hp.nside2pixarea(nside) / 4. / np.pi * np.sum(msk**2.)**2. / np.sum(msk**4.)
print("fsky:", fsky)
fsky_eff = hp.nside2pixarea(nside) / 4. / np.pi * np.sum(msk_inv)**2. / np.sum(msk_inv**2.)
print("fsky_eff:", fsky_eff)
fsky_inv = hp.nside2pixarea(nside) / 4. / np.pi * np.sum(msk_inv**2.)**2. / np.sum(msk_inv**4.)
print("fsky_inv:", fsky_inv)
# exit()

bins, leff = cf.setup_bins(nside, lmax_o=lmax_o, bsz_0=30, fixed_bins=True)
ells = np.arange(lmax+1)
Dell_factor = ells * (ells + 1.) / 2. / np.pi
# def Dl_binner(cl_input, is_Cell=True):
#     if is_Cell:
#         Dl = cl_input[:lmax+1] * Dell_factor
#     else:
#         Dl = cl_input[:lmax+1]
#     Dl_bin = cf.binner(Dl, lmax_o, 30, leff, is_Cell = False, fixed_bins=True)
#     return Dl, Dl_bin
# DlTT, DlTT_bin = Dl_binner(cl_tt)
# DlEE, DlEE_bin = Dl_binner(cl_ee)
# DlTE, DlTE_bin = Dl_binner(cl_te)
# DlBB, DlBB_bin = Dl_binner(cl_bb_lensed)

def compute_Dell(Bmap, beam=11., TEB=2, msk_apo=msk_20c2):
    fld_B = nmt.NmtField(msk_apo, [Bmap], lmax_sht=lmax, masked_on_input=False)
    B_wsp = nmt.NmtWorkspace()
    B_wsp.compute_coupling_matrix(fld_B, fld_B, bins)

    Cl_coup_BB = nmt.compute_coupled_cell(fld_B, fld_B)/hp.gauss_beam(np.deg2rad(beam / 60.), pol=True, lmax=lmax)[:,TEB]**2
    Dl_BB_nmt = B_wsp.decouple_cell(Cl_coup_BB)[0]
    return Dl_BB_nmt
def compute_cross_Dell(Bmap1, Bmap2, beam=11., TEB1=0, TEB2=1, msk_apo=msk_20c2):
    fld_B1 = nmt.NmtField(msk_apo, [Bmap1], lmax_sht=lmax, masked_on_input=False)
    fld_B2 = nmt.NmtField(msk_apo, [Bmap2], lmax_sht=lmax, masked_on_input=False)
    B_wsp = nmt.NmtWorkspace()
    B_wsp.compute_coupling_matrix(fld_B1, fld_B2, bins)

    Cl_coup_BB = nmt.compute_coupled_cell(fld_B1, fld_B2)/hp.gauss_beam(np.deg2rad(beam / 60.), pol=True, lmax=lmax)[:,TEB1]/hp.gauss_beam(np.deg2rad(beam / 60.), pol=True, lmax=lmax)[:,TEB2]
    Dl_BB_nmt = B_wsp.decouple_cell(Cl_coup_BB)[0]
    return Dl_BB_nmt
def fetch_cmb(sim):
    file_name = 'LensInfo/CMB_LEN.fits'
    file_path = os.path.join(lens_root, str(sim).zfill(3), file_name)
    cmb = hp.read_map(file_path, field=None, dtype=np.float64)
    # cmb = cmb * mask_alicpt
    return cmb
cmb_iqu = fetch_cmb(0)
# Bcmb = cf.get_cleanedBmap(cmb_iqu, msk_20, lmax_sht=lmax_rot)
# DlBB_cmb = compute_Dell(Bcmb, 0., 2)
# TEmap = cf.iqu2teb(cmb_iqu, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
# DlTT_cmb = compute_Dell(TEmap[0], 0., 0)
# DlEE_cmb = compute_Dell(TEmap[1], 0., 1)
# DlTE_cmb = compute_cross_Dell(TEmap[0], TEmap[1], 0., 0, 1)

def gridlines():
    f = plt.gcf()
    cbax = f.get_children()[2] # color bar axis
    # print(cbax.get_children())
    coord_text_obj = cbax.get_children()[2] # text bar, found by the last print step
    # print(dir(coord_text_obj))
    coord_text_obj.set_fontsize(10)
    # print(coord_text_obj.get_position())
    coord_text_obj.set_position((0.5,-1.2))
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
for freq in freqs:
    fwhm = fwhms[freq]

    cmb_smooth = hp.smoothing(cmb_iqu, fwhm=np.deg2rad(fwhm / 60.), use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')

    IQU = hp.read_map(lens_root+str(sim).zfill(3)+'/Ali_'+freq+'.fits', field=None, dtype=np.float64)
    Bmap_bs = cf.get_cleanedBmap(IQU, msk_20, lmax_sht=lmax_rot)
    # cf.plot_maps(Bmap_bs*msk_20c2, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-1., vmax=1., resol='print', show=False)
    # gridlines()
    # plt.savefig(output_root+'Bcmbpfg_beamsys_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)
    DlBB_bs = compute_Dell(Bmap_bs, fwhm, 2)
    TEmap = cf.iqu2teb(IQU, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
    DlTT_bs = compute_Dell(TEmap[0], fwhm, 0)
    DlEE_bs = compute_Dell(TEmap[1], fwhm, 1)
    DlTE_bs = compute_cross_Dell(TEmap[0], TEmap[1], fwhm, 0, 1)

    IQU = hp.read_map(lens_root+str(sim).zfill(3)+'/DEPROJ_Ali_'+freq+'.fits', field=None, dtype=np.float64)
    Bmap_dp = cf.get_cleanedBmap(IQU, msk_20, lmax_sht=lmax_rot)
    # cf.plot_maps(Bmap_dp*msk_20c2, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-1., vmax=1., resol='print', show=False)
    # gridlines()
    # plt.savefig(output_root+'Bcmbpfg_deproj_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)
    DlBB_dp = compute_Dell(Bmap_dp, fwhm, 2)
    TEmap = cf.iqu2teb(IQU, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
    DlTT_dp = compute_Dell(TEmap[0], fwhm, 0)
    DlEE_dp = compute_Dell(TEmap[1], fwhm, 1)
    DlTE_dp = compute_cross_Dell(TEmap[0], TEmap[1], fwhm, 0, 1)
    TEmap_dp = TEmap

    fg_iqu = hp.read_map(fg_root+'group1_map_'+freq+'GHz.fits', field=None, dtype=np.float64)
    IQU = cmb_smooth + fg_iqu
    Bmap_wo = cf.get_cleanedBmap(IQU, msk_20, lmax_sht=lmax_rot)
    # cf.plot_maps(Bmap_wo*msk_20c2, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-1., vmax=1., resol='print', show=False)
    # gridlines()
    # plt.savefig(output_root+'Bcmbpfg_wobs_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)
    DlBB_wo = compute_Dell(Bmap_wo, fwhm, 2)
    TEmap = cf.iqu2teb(IQU, mask_in=msk_20, teb='te', lmax_sht=lmax_rot, return_alm=False)
    DlTT_wo = compute_Dell(TEmap[0], fwhm, 0)
    DlEE_wo = compute_Dell(TEmap[1], fwhm, 1)
    DlTE_wo = compute_cross_Dell(TEmap[0], TEmap[1], fwhm, 0, 1)
    TEmap_wo = TEmap
    # cf.plot_maps((Bmap_dp-Bmap_wo)*msk_20c2, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-1., vmax=1., resol='print', show=False)
    # gridlines()
    # plt.savefig(output_root+'Bcmbpfg_res_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)
    # cf.plot_maps((TEmap_dp[0]-TEmap_wo[0])*msk_20c2, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-1., vmax=1., resol='print', show=False)
    # gridlines()
    # plt.savefig(output_root+'Tcmbpfg_res_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)
    # cf.plot_maps((TEmap_dp[1]-TEmap_wo[1])*msk_20c2, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', vmin=-1., vmax=1., resol='print', show=False)
    # gridlines()
    # plt.savefig(output_root+'Ecmbpfg_res_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)

    del Bmap_wo, Bmap_dp, Bmap_bs, IQU, TEmap
    np.savetxt(output_root+'Dl_'+freq+'_bsz'+str(bin_size)+'.dat', [leff, DlTT_wo, DlTT_bs, DlTT_dp,
                                                                    DlEE_wo, DlEE_bs, DlEE_dp,
                                                                    DlBB_wo, DlBB_bs, DlBB_dp,
                                                                    DlTE_wo, DlTE_bs, DlTE_dp])

for freq in freqs:
    leff, DlTT_wo, DlTT_bs, DlTT_dp,\
    DlEE_wo, DlEE_bs, DlEE_dp,\
    DlBB_wo, DlBB_bs, DlBB_dp,\
    DlTE_wo, DlTE_bs, DlTE_dp = np.loadtxt(output_root+'Dl_'+freq+'_bsz'+str(bin_size)+'.dat')  

    fig, axes = plt.subplots(2, 2, figsize=(3.5*2, 2.16*2), dpi=600, sharex='all')
    fig.subplots_adjust(hspace=.07, wspace=0.3)
    plt.style.use('seaborn-v0_8-ticks')
    plt.rc('font', family='sans-serif', size=6)
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    ax = axes[1,1]
    # ax.plot(ell[30:], DlBB_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
    # ax.plot(leff[1:], (DlBB_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
    ax.plot(leff[1:], DlBB_wo[1:], 'k-', lw=1., alpha=0.7)
    ax.plot(leff[1:], DlBB_bs[1:], 'b-', lw=1., alpha=0.7)
    ax.plot(leff[1:], DlBB_dp[1:], 'ro-', lw=1., alpha=0.7, ms=1)
    ax.plot(leff[1:], np.abs(DlBB_dp - DlBB_wo)[1:], 'c--', lw=1., alpha=0.7)
    # ax.plot(leff[1:], (DlBB_fg)[1:], '--', lw=1., alpha=0.7, label='BS.FG. '+freq+'GHz')
    ax.text(0.05, 0.95, r'$BB$', ha='left', va='top', transform=ax.transAxes, fontdict=dict(fontsize=7))
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e-7, ymax=1.e2)
    # ax.grid(which='both', axis='both')

    ax = axes[0,1]
    # ax.plot(ell[30:], DlEE_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
    # ax.plot(leff[1:], (DlEE_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
    ax.plot(leff[1:], DlEE_wo[1:], 'k-', lw=1.)
    ax.plot(leff[1:], DlEE_bs[1:], 'b-', lw=1., alpha=0.7)
    ax.plot(leff[1:], DlEE_dp[1:], 'ro-', lw=1., alpha=0.7, ms=1)
    ax.plot(leff[1:], np.abs(DlEE_dp - DlEE_wo)[1:], 'c--', lw=1., alpha=0.7)
    # ax.plot(leff[1:], (DlEE_fg)[1:], '--', lw=1., alpha=0.7, label='BS.FG. '+freq+'GHz')
    ax.text(0.05, 0.95, r'$EE$', ha='left', va='top', transform=ax.transAxes, fontdict=dict(fontsize=7))
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{EE}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e-7, ymax=1.e2)
    # ax.grid(which='both', axis='both')

    ax = axes[0,0]
    # ax.plot(ell[30:], DlTT_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
    # ax.plot(leff[1:], (DlTT_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
    ax.plot(leff[1:], DlTT_wo[1:], 'k-', lw=1., label='input CMB + FG.')
    ax.plot(leff[1:], DlTT_bs[1:], 'b-', lw=1., alpha=0.7, label='BS. CMB + FG.')
    ax.plot(leff[1:], DlTT_dp[1:], 'ro-', lw=1., alpha=0.7, ms=1, label='Deproj. CMB + FG.')
    ax.plot(leff[1:], np.abs(DlTT_dp - DlTT_wo)[1:], 'c--', lw=1., alpha=0.7, label='Deproj. residual')
    # ax.plot(leff[1:], (DlTT_fg)[1:], '--', lw=1., alpha=0.7, label='BS.FG. '+freq+'GHz')
    ax.text(0.05, 0.55, freq+' GHz', ha='left', va='top', transform=ax.transAxes, fontdict=dict(fontsize=9), bbox={'alpha':0.1, 'pad': 0.8})
    ax.text(0.05, 0.95, r'$TT$', ha='left', va='top', transform=ax.transAxes, fontdict=dict(fontsize=7))
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{TT}_\ell$ [in $\mu$K${}^2$]')
    ax.legend(loc='best', frameon=False, fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e-7, ymax=1.e2)
    # ax.grid(which='both', axis='both')

    ax = axes[1,0]
    # ax.plot(ell[30:], DlTE_lensed[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='lensed')
    # ax.plot(leff[1:], (DlTE_bin)[1:], 'k--', lw=1., alpha=0.7, label='theo.CMB')
    ax.plot(leff[1:], DlTE_wo[1:], 'k-', lw=1.)
    ax.plot(leff[1:], DlTE_bs[1:], 'b-', lw=1., alpha=0.7)
    ax.plot(leff[1:], DlTE_dp[1:], 'ro-', lw=1., alpha=0.7, ms=1)
    ax.plot(leff[1:], (DlTE_dp - DlTE_wo)[1:], 'c--', lw=1., alpha=0.7)
    # ax.plot(leff[1:], (DlTT_fg)[1:], '--', lw=1., alpha=0.7, label='BS.FG. '+freq+'GHz')
    ax.text(0.05, 0.95, r'$TE$', ha='left', va='top', transform=ax.transAxes, fontdict=dict(fontsize=7))
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{TE}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=7)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e-7, ymax=1.e2)
    # ax.grid(which='both', axis='both')
    plt.savefig(output_root+'Dl_'+freq+'GHz_bsz'+str(bin_size)+'.png',bbox_inches='tight',pad_inches=0.1)

# plt.show()







