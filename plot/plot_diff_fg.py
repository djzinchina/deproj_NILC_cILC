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
fwhms = {'95':19., '150':11., '143':7.303, '100':9.682}
temps = {'95':'100', '150':'143'}
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
# fsky_eff = hp.nside2pixarea(nside) / 4. / np.pi * np.sum(msk_inv)**2. / np.sum(msk_inv**2.)
# print("fsky_eff:", fsky_eff)
# fsky_inv = hp.nside2pixarea(nside) / 4. / np.pi * np.sum(msk_inv**2.)**2. / np.sum(msk_inv**4.)
# print("fsky_inv:", fsky_inv)
# exit()

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

def rebeam(map_in, fwhm_in, fwhm_out):
    alm_in = hp.map2alm(map_in, lmax=lmax_rot, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
    beam_in = hp.gauss_beam(np.deg2rad(fwhm_in / 60.), pol=True, lmax=lmax_rot)
    beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), pol=True, lmax=lmax_rot)
    beam_ratio = cf.compute_beam_ratio(beam_in[:,2], beam_out[:,2])
    alm_out = hp.almxfl(alm_in, beam_ratio)
    map_out = hp.alm2map(alm_out, nside)
    return map_out

for freq in freqs:
    fwhm = fwhms[freq]

    cmb_smooth = hp.smoothing(cmb_iqu, fwhm=np.deg2rad(fwhm / 60.), use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')


    fg_iqu = hp.read_map(fg_root+'group1_map_'+freq+'GHz.fits', field=None, dtype=np.float64)

    cf.plot_maps(cmb_smooth[0], mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', resol='print', show=False)
    gridlines()
    plt.savefig(output_root+'Tcmb_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)

    cf.plot_maps(fg_iqu[0], mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', resol='print', show=False)
    gridlines()
    plt.savefig(output_root+'Tfg_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)

    fg_temp = hp.read_map(fg_root+'group1_map_'+temps[freq]+'GHz.fits', field=None, dtype=np.float64)
    fg_temp_re = rebeam(fg_temp[0], fwhms[temps[freq]], fwhm)
    cf.plot_maps(fg_iqu[0] - fg_temp_re, mask_in=msk_20, title=None, proj='orth', unit=r'$\mu$K', resol='print', show=False)
    gridlines()
    plt.savefig(output_root+'Tdifffg_'+freq+'GHz.png', bbox_inches='tight',pad_inches=0.)










