import numpy as np
import healpy as hp 
import os
import cmbframe as cf
import pymaster as nmt
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from tqdm import tqdm  
import nilc_weights2 as nw
import chilc

r = 0.0
nside = 1024
lmax = 2 * nside 
lmax_o = 1500
start_sim = 0
nsims = 300 # set 0 to avoid re-run again
show = False
map_fwhm = 11. # in arcmin

#============ Folder Parameters ==============

foreground_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/BeamSys/FG/0' # with BeamSys
fg_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/WOBeamSys/FG/' # WO Beamsys
# cmb_root = '/media/doujzh/AliCPT_data/LensedCMB'
instrs = ['WMAP', 'Ali', 'HFI', 'HFI', 'Ali','HFI', 'HFI']
freqs_ = ['K', '95', '100', '143', '150', '217', '353']
freqs = ['23', '95', '100', '143', '150', '217', '353']
lens_root = '/media/doujzh/AliCPT_data/Zirui_beamsys/LensWithBeamSys'
noise_root = '/media/doujzh/Ancillary_data/cNILC_covariance_matrix/resources/Noise_sims/'
output_root = '/media/doujzh/AliCPT_data/AliCPT_lens2'

mask_alicpt = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_survey_mask_C_1024.fits',field=0, dtype=np.float64)
mask_30 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_30uKcut150_C_1024.fits', field=None, dtype=np.float64)
mask_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64)
mask_UNP = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=np.float64)

mask_c2 = nmt.mask_apodization(mask_20, 6., "C2")
mask_UNP_c2 = nmt.mask_apodization(mask_UNP, 6., "C2")
mask_apo = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64)

fsky = hp.nside2pixarea(nside) / 4. / np.pi * np.sum(mask_alicpt**2.)**2. / np.sum(mask_alicpt**4.)
map_sel = np.arange(len(freqs_))
bands_beam = [52.8, 19., 9.682, 7.303, 11., 5.021, 4.944]

lmax_ch = np.array([350, 1200, lmax, lmax, lmax, lmax, lmax])

nu_dim = len(map_sel)

Rot = hp.Rotator(coord=('G', 'C'))
fg = []
beams = []
for nu in map_sel:
    # file_name = instrs[nu]+'_'+freqs_[nu]+'.fits'
    # file_path = os.path.join(foreground_root, file_name)
    # fg.append(hp.read_map(file_path, field=None, dtype=np.float64) * mask_alicpt)

    fg_iqu = hp.read_map(fg_root+'group1_map_'+freqs[nu]+'GHz.fits', field=None, dtype=np.float64)
    fg.append(fg_iqu * mask_alicpt)

    beams.append(hp.gauss_beam(np.deg2rad(bands_beam[nu] / 60.), pol=True, lmax=lmax))
fg = np.array(fg)
# exit()
beams = np.array(beams)
beam_0 = hp.gauss_beam(np.deg2rad(map_fwhm / 60.), pol=True, lmax=lmax)

def fetch_cmbpfg(sim, deproj=True):
    cmbpfg = []
    for nu in map_sel:
        file_name = instrs[nu]+'_'+freqs_[nu]+'.fits'
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

def smooth_cmb(cmb):
    cmb_smooth = []
    for nu in map_sel:
        cmb_sm = hp.smoothing(cmb, fwhm=np.deg2rad(bands_beam[nu] / 60.), use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
        cmb_smooth.append(cmb_sm * mask_alicpt)
        del cmb_sm

    return cmb_smooth
# cmb0 = fetch_cmb(0)
# cf.plot_maps(cmb0[0], proj='moll', outfile=output_root+'/maps/TCMB.png', show=False)
# exit()
def fetch_noise(sim):
    noise_path = noise_root+str(sim)+'/'
    noise_out = []
    for nu in map_sel:
        noise_filename = noise_path + 'Noise_IQU_'+ freqs_[nu] +'.fits'
        noise_out.append(hp.read_map(noise_filename, field=None, dtype=np.float64, verbose=False)*mask_alicpt)
    # noise_out = np.array(noise_out)
    return noise_out

#========== Setup wwavelets ==============
band_lims = [15,30,60,120,300,700,1200,lmax]
# band_lims = [15,45,75,105,150,210,250,350,500,700,lmax]
# band_fwhms = [300.,120., 60., 45., 30., 15., 10., 7.5, 5.]
# band_fwhms = [300., 120., 45., 30., 15., 7.5, 5.]
# band_fwhms = [300., 150., 90., 75., 40., 20., 11.]

bands = cf.cosine_bands(band_lims)
# bands = cf.gaussian_bands(band_fwhms, lmax_band=lmax)
# cf.plot_needlet_bands(bands, file_out=output_root+'/nilc_bands.png', show=False) #xscale="log",

nbands = len(bands[0,:])

Tbeam_ratios = cf.beam_ratios(beams[:,:,0], beam_0[:,0])
Ebeam_ratios = cf.beam_ratios(beams[:,:,1], beam_0[:,1])

bins_BB, leff_BB, bsz_BB, lmin_BB = cf.setup_bins(nside, lmax_o)
bins, leff = cf.setup_bins(nside, lmax_o=lmax_o, bsz_0=30, fixed_bins=True)

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

for sim in tqdm(range(start_sim, start_sim + nsims), ncols=120):
    # fno, fig, ax = cf.make_plotaxes()
    # ax.plot(ells[40:lmax_o+1], beam_0[40:lmax_o+1,0], '-', label='T')
    # ax.plot(ells[40:lmax_o+1], beam_0[40:lmax_o+1,1], '-', label='E')
    # ax.plot(ells[40:lmax_o+1], beam_0[40:lmax_o+1,2], '-', label='B')
    # ax.set_xlabel(r'$\ell$')
    # ax.set_ylabel(r'$b_\ell$')
    # ax.legend(loc='best', frameon=False, fontsize=12)
    # plt.savefig(output_root+'/test_beam.png',bbox_inches='tight',pad_inches=0.1)
    # exit()
    # cmb = fetch_cmb(sim)
    # cl_cmb = hp.anafast(cmb, nspec=4, lmax=lmax, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')# / fsky
    # fno, fig, ax = cf.make_plotaxes()
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_cmb[0, 40:lmax_o+1], '-', label='TT anafast')
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_tt[40:lmax_o+1], '-', label='TT input')
    # ax.set_xlabel(r'$\ell$')
    # ax.set_ylabel(r'$\mathcal{D}^{TT}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=12)
    # plt.savefig(output_root+'/test_TT.png',bbox_inches='tight',pad_inches=0.1)
    # fno, fig, ax = cf.make_plotaxes()
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_cmb[1, 40:lmax_o+1], '-', label='EE anafast')
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_ee[40:lmax_o+1], '-', label='EE input')
    # ax.set_xlabel(r'$\ell$')
    # ax.set_ylabel(r'$\mathcal{D}^{EE}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=12)
    # plt.savefig(output_root+'/test_EE.png',bbox_inches='tight',pad_inches=0.1)
    # fno, fig, ax = cf.make_plotaxes()
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_cmb[2, 40:lmax_o+1], '-', label='BB anafast')
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_bb[40:lmax_o+1], '-', label='BB input')
    # ax.set_xlabel(r'$\ell$')
    # ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=12)
    # plt.savefig(output_root+'/test_BB.png',bbox_inches='tight',pad_inches=0.1)
    # fno, fig, ax = cf.make_plotaxes()
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_cmb[3, 40:lmax_o+1], '-', label='TE anafast')
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_te[40:lmax_o+1], '-', label='TE input')
    # ax.set_xlabel(r'$\ell$')
    # ax.set_ylabel(r'$\mathcal{D}^{TE}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=12)
    # plt.savefig(output_root+'/test_TE.png',bbox_inches='tight',pad_inches=0.1)
    # exit()
    # cmb_sim = hp.smoothing(cmb, fwhm=np.deg2rad(map_fwhm / 60.), use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
    # cmbpfg_sim = fetch_cmbpfg(sim, deproj=True)
    # noise_sim = fetch_noise(sim)
    # map_sim = np.array(cmbpfg_sim) + np.array(noise_sim)
    # dr_sim = np.array(cmbpfg_sim) - np.array(smooth_cmb(cmb)) - fg
    # print(dr_sim.shape, fg.shape)
    # if sim == 0: cf.plot_maps(dr_sim[2][2], mask_in=mask_alicpt, proj='orth',  outfile=output_root+'/Zero_100GHz_sim'+str(sim)+'.png', show=False)
    # exit()
    # Blm_wgt = []
    # Blm_sim = []
    # Blmapo_sim = []
    # Nlm_sim = []
    # for nu in map_sel:
        # cf.plot_maps(map_sim[-1][0], vmin=-450., vmax=450., proj='orth', outfile=output_root+'/maps/input_nu'+str(nu)+'_sim'+str(sim)+'.png', show=False)

        # Bmap_nu = cf.get_cleanedBmap(map_sim[nu], mask_UNP, lmax_sht=lmax)

        # if nu == 4: Bmap_act = cf.get_cleanedBmap(cmb_sim[4], mask_UNP, lmax_sht=lmax)
        # Blm_wgt.append(hp.map2alm(Bmap_nu * mask_apo, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts'))
        # Blm_sim.append(hp.map2alm(Bmap_nu * mask_UNP, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts'))
        # Blmapo_sim.append(hp.map2alm(Bmap_nu * mask_apo, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts'))

        # Nmap_nu = cf.get_cleanedBmap(noise_sim[nu], mask_UNP, lmax_sht=lmax) 
        # Nlm_sim.append(hp.map2alm(Nmap_nu * mask_UNP, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts'))
        # del Bmap_nu , Nmap_nu
    # Bmap_act = cf.get_cleanedBmap(cmb_sim, mask_UNP, lmax_sht=lmax)


    # Blm_sim = np.array(Blm_sim)
    # Blmapo_sim = np.array(Blmapo_sim)
    # Nlm_sim = np.array(Nlm_sim)

    # Blm_dr_sim = np.zeros_like(Blmapo_sim)
    # for nu in [1, 4]:
    #     Bdr_nu = cf.get_cleanedBmap(dr_sim[nu], mask_UNP, lmax_sht=lmax)
    #     Blm_dr_sim[nu] = hp.map2alm(Bdr_nu * mask_UNP, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')


    # wavelet_Tmaps = []
    # wavelet_Emaps = []
    # for nu in tqdm(range(nu_dim), ncols=120):
    #     TEmap = cf.iqu2teb(map_sim[nu], mask_in=mask_30, teb='te', lmax_sht=lmax, return_alm=False) #*ps_msk

    #     # if nu == 4: TEmap_act = cf.iqu2teb(cmb_sim[4], mask_in=mask_30, teb='te', lmax_sht=lmax, return_alm=False)

    #     Tlms = hp.map2alm(TEmap[0]*mask_30, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
    #     beam_adjd_Tlm = hp.almxfl(Tlms, Tbeam_ratios[nu])

    # # =============================================================
    #     Elms = hp.map2alm(TEmap[1]*mask_30, lmax=lmax, pol=False, use_pixel_weights=True, datapath='/home/doujzh/DATA/HPX_pix_wgts')
    #     beam_adjd_Elm = hp.almxfl(Elms, Ebeam_ratios[nu])

    #     wavelet_Tmaps.append(cf.alm2wavelet(beam_adjd_Tlm, bands, w_nside_max=nside))
    #     wavelet_Emaps.append(cf.alm2wavelet(beam_adjd_Elm, bands, w_nside_max=nside))
    # TEmap_act = cf.iqu2teb(cmb_sim, mask_in=mask_30, teb='te', lmax_sht=lmax, return_alm=False)
    
    # nilc_Tmap_wav = []
    # nilc_Emap_wav = []


    # nilc_Twgt_byband = []
    # nilc_Ewgt_byband = []

    # nside_prev = -1
    # nu_arr = np.arange(nu_dim, dtype=np.int_)
    # for band in tqdm(range(nbands), ncols=120):

    #     lmax_band = cf.get_lmax_band(bands[:,band])

    #     # print(lmax_band, nu_arr[lmax_ch >= lmax_band])

    #     # print(band)
    #     Tnus_in_band = []
    #     Enus_in_band = []
    #     for nu in nu_arr[lmax_ch >= lmax_band]:
    #         Tnus_in_band.append(wavelet_Tmaps[nu][band])
    #         Enus_in_band.append(wavelet_Emaps[nu][band])

    #     nside_band = hp.npix2nside(len(Tnus_in_band[0]))
    #     Twsp = nw.nilc_weights_new(Tnus_in_band)
    #     Ewsp = nw.nilc_weights_new(Enus_in_band)

    #     T_nilc_wgts = Twsp.get_weights()
    #     E_nilc_wgts = Ewsp.get_weights()

    #     # print(T_nilc_wgts.shape)
    #     # np.savetxt(output_root+'/data/Tnilcweights_band'+str(band)+'_sim'+str(sim)+'.dat', T_nilc_wgts)
    #     # np.savetxt(output_root+'/data/Enilcweights_band'+str(band)+'_sim'+str(sim)+'.dat', E_nilc_wgts)

    #     for nu in range(np.sum(lmax_ch >= lmax_band)):
    #         if nu == 0:
    #             nilc_T_band = T_nilc_wgts[nu] * Tnus_in_band[nu]
    #             nilc_E_band = E_nilc_wgts[nu] * Enus_in_band[nu]

    #         else :
    #             nilc_T_band += T_nilc_wgts[nu] * Tnus_in_band[nu]
    #             nilc_E_band += E_nilc_wgts[nu] * Enus_in_band[nu]

    #     nilc_Tmap_wav.append(nilc_T_band)
    #     nilc_Emap_wav.append(nilc_E_band)

    #     # cf.plot_needlet_maps(T_nilc_wgts, proj='orth', outfile=output_root+'/maps/NILC_Twgts_band'+str(band)+'_sim'+str(sim)+'.png', show=False)
    #     # cf.plot_needlet_maps(E_nilc_wgts, proj='orth', outfile=output_root+'/maps/NILC_Ewgts_band'+str(band)+'_sim'+str(sim)+'.png', show=False)

    #     nilc_Twgt_byband.append(T_nilc_wgts)
    #     nilc_Ewgt_byband.append(E_nilc_wgts)

    #     # nside_prev = nside_band

    #     del Enus_in_band, Ewsp, E_nilc_wgts, nilc_E_band, nilc_T_band, Tnus_in_band, Twsp, T_nilc_wgts

    # B_wsp = chilc.cilc_cleaner(Blmapo_sim, beams=beams[:,:,2], com_res_beam=beam_0[:,2])
    
    # print("Computing cILC weights")
    # B_wsp.compute_cilc_weights(map_sel, bandpass='real', lmax_ch=lmax_ch)
    # del Blmapo_sim
    # ilc_wgts = np.reshape(B_wsp.har_wgts.T, (lmax+1, nu_dim,1))  # shape = (lmax+1, nu_dim, 1)
    # np.savetxt(output_root+'/data/cILC_wgts_B_'+str(sim).zfill(3)+'.dat', B_wsp.har_wgts)
    # cf.plot_ilc_weights(B_wsp.har_wgts, 'cILC', label_list=['K','95','100','143','150','217','353'], outfile=output_root+'/maps/cILC_Bwgts_sim'+str(sim)+'.png', show=False)
    # print("Computing T NILC weights and map")
    # T_nilc = cf.wavelet2map(nside, nilc_Tmap_wav, bands) * mask_20
    # print("Computing E NILC weights and map")
    # E_nilc = cf.wavelet2map(nside, nilc_Emap_wav, bands) * mask_20

    # print("Getting cILC B map")
    # B_cilc = hp.alm2map(B_wsp.get_projected_alms(Blm_sim), nside, pol=False) * mask_UNP

    # B_dr = hp.alm2map(B_wsp.get_projected_alms(Blm_dr_sim), nside, pol=False) # deproj. residual
    # hp.write_map(output_root+'/maps/deproj-residual_Bcilc_11arcmin_sim'+str(sim)+'.fits', B_dr, overwrite=True)
    # B_dr = hp.read_map(output_root+'/maps/deproj-residual_Bcilc_11arcmin_sim'+str(sim)+'.fits', field=None, dtype=np.float64) * mask_UNP
    
    # cf.plot_maps(T_nilc, mask_in=mask_20, proj='orth',  outfile=output_root+'/maps/Tnilc_sim'+str(sim)+'.png', show=False)
    # cf.plot_maps(E_nilc, mask_in=mask_20, proj='orth',  outfile=output_root+'/maps/Enilc_sim'+str(sim)+'.png', show=False)
    # cf.plot_maps(B_cilc, mask_in=mask_UNP, proj='orth', outfile=output_root+'/maps/Bcilc_sim'+str(sim)+'.png', show=False)

    # cf.plot_maps(T_nilc - TEmap_act[0], mask_in=mask_20, proj='orth',  outfile=output_root+'/maps/Tres_sim'+str(sim)+'.png', show=False)
    # cf.plot_maps(E_nilc - TEmap_act[1], mask_in=mask_20, proj='orth',  outfile=output_root+'/maps/Eres_sim'+str(sim)+'.png', show=False)
    # cf.plot_maps(B_cilc - Bmap_act,     mask_in=mask_UNP, proj='orth', outfile=output_root+'/maps/Bres_sim'+str(sim)+'.png', show=False)

    # hp.write_map(output_root+'/maps/TEnilc-Bcilc_11arcmin_sim'+str(sim)+'.fits', [T_nilc, E_nilc, B_cilc], overwrite=True)
    # hp.write_map(output_root+'/maps/tot-residual_TEnilc-Bcilc_11arcmin_sim'+str(sim)+'.fits', \
    #     [(T_nilc - TEmap_act[0]) * mask_20, (E_nilc - TEmap_act[1]) * mask_20, (B_cilc - Bmap_act) * mask_UNP], overwrite=True)
    # T_res = (T_nilc - TEmap_act[0]) * mask_20
    # E_res = (E_nilc - TEmap_act[1]) * mask_20
    # B_res = (B_cilc - Bmap_act) * mask_UNP

    # wavelet_Tnoise = []
    # wavelet_Enoise = []
    # for nu in tqdm(range(nu_dim), ncols=120, leave=False):
    #     TEnoise = cf.iqu2teb(noise_sim[nu], mask_in=mask_30, teb='te', lmax_sht=lmax, return_alm=True) #*ps_msk

    #     beam_adjd_Nlm_T = hp.almxfl(TEnoise[0], Tbeam_ratios[nu])
    #     beam_adjd_Nlm_E = hp.almxfl(TEnoise[1], Ebeam_ratios[nu])

    #     wavelet_Tnoise.append(cf.alm2wavelet(beam_adjd_Nlm_T, bands, w_nside_max=nside))
    #     wavelet_Enoise.append(cf.alm2wavelet(beam_adjd_Nlm_E, bands, w_nside_max=nside))

    # projTN_wav = []
    # projEN_wav = []

    # for band in tqdm(range(nbands), ncols=120, leave=False):

    #     lmax_band = cf.get_lmax_band(bands[:,band])

    #     # print(lmax_band, nu_arr[lmax_ch >= lmax_band])

    #     # print(band)
    #     Tnus_in_band = []
    #     Enus_in_band = []
    #     for nu in nu_arr[lmax_ch >= lmax_band]:
    #         Tnus_in_band.append(wavelet_Tnoise[nu][band])
    #         Enus_in_band.append(wavelet_Enoise[nu][band])

    #     for nu in range(np.sum(lmax_ch >= lmax_band)):
    #         if nu == 0:
    #             nilc_T_band = nilc_Twgt_byband[band][nu] * Tnus_in_band[nu]
    #             nilc_E_band = nilc_Ewgt_byband[band][nu] * Enus_in_band[nu]

    #         else :
    #             nilc_T_band += nilc_Twgt_byband[band][nu] * Tnus_in_band[nu]
    #             nilc_E_band += nilc_Ewgt_byband[band][nu] * Enus_in_band[nu]

    #     projTN_wav.append(nilc_T_band)
    #     projEN_wav.append(nilc_E_band)

    #     del Tnus_in_band, nilc_T_band, Enus_in_band, nilc_E_band

    # T_nilc_noise = cf.wavelet2map(nside, projTN_wav, bands) * mask_20
    # E_nilc_noise = cf.wavelet2map(nside, projEN_wav, bands) * mask_20
    # B_cilc_noise = hp.alm2map(B_wsp.get_projected_alms(Nlm_sim), nside, pol=False) * mask_UNP

    # hp.write_map(output_root+'/maps/TEnilc-Bcilc_proj-noise_11arcmin_sim'+str(sim)+'.fits', [T_nilc_noise, E_nilc_noise, B_cilc_noise], overwrite=True)
    # del map_sim, Blm_sim, B_wsp, Nlm_sim, noise_sim

    clean_maps = hp.read_map(output_root+'/maps/TEnilc-Bcilc_11arcmin_sim'+str(sim)+'.fits', field=None, dtype=np.float64)
    T_nilc, E_nilc, B_cilc = clean_maps[0], clean_maps[1], clean_maps[2]
    res_maps = hp.read_map(output_root+'/maps/tot-residual_TEnilc-Bcilc_11arcmin_sim'+str(sim)+'.fits', field=None, dtype=np.float64)
    T_res, E_res, B_res = res_maps[0], res_maps[1], res_maps[2]
    res_noise = hp.read_map(output_root+'/maps/TEnilc-Bcilc_proj-noise_11arcmin_sim'+str(sim)+'.fits', field=None, dtype=np.float64)
    T_nilc_noise, E_nilc_noise, B_cilc_noise = res_noise[0], res_noise[1], res_noise[2]

    # B_res = hp.read_map(output_root+'/maps/tot-residual_TEnilc-Bcilc_11arcmin_sim'+str(sim)+'.fits', field=None, dtype=np.float64)[2]
    # B_cilc_noise = hp.read_map(output_root+'/maps/TEnilc-Bcilc_proj-noise_11arcmin_sim'+str(sim)+'.fits', field=None, dtype=np.float64)[2]

    Nl_coup_TT = cf.map2coupCl_nmt(T_nilc_noise, mask_c2, beam=beam_0[:,0], lmax_sht=lmax)
    Nl_coup_EE = cf.map2coupCl_nmt(E_nilc_noise, mask_c2, beam=beam_0[:,1], lmax_sht=lmax)
    Nl_coup_BB = cf.map2coupCl_nmt(B_cilc_noise, mask_apo, beam=beam_0[:,2], lmax_sht=lmax, masked_on_input=False)

    if sim == start_sim:
        Dl_TT, Twsp_nmt = cf.map2Cl_nmt(T_nilc, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,0], noise_Cl=Nl_coup_TT, return_wsp=True) 
        Dl_EE, Ewsp_nmt = cf.map2Cl_nmt(E_nilc, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,1], noise_Cl=Nl_coup_EE, return_wsp=True) 
        Dl_BB, Bwsp_nmt = cf.map2Cl_nmt(B_cilc, mask_apo, bins, lmax_sht=lmax, beam=beam_0[:,2], noise_Cl=Nl_coup_BB, masked_on_input=False, return_wsp=True)
        Dl_TE, Xwsp_nmt = cf.map2Cl_nmt(T_nilc, mask_c2, bins, map_in2=E_nilc, lmax_sht=lmax, beam=beam_0[:,0], beam2=beam_0[:,1], return_wsp=True)
    else:
        Dl_TT = cf.map2Cl_nmt(T_nilc, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,0], noise_Cl=Nl_coup_TT, reuse_wsp=Twsp_nmt)
        Dl_EE = cf.map2Cl_nmt(E_nilc, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,1], noise_Cl=Nl_coup_EE, reuse_wsp=Ewsp_nmt)
        Dl_BB = cf.map2Cl_nmt(B_cilc, mask_apo, bins, lmax_sht=lmax, beam=beam_0[:,2], noise_Cl=Nl_coup_BB, masked_on_input=False, reuse_wsp=Bwsp_nmt)
        Dl_TE = cf.map2Cl_nmt(T_nilc, mask_c2, bins, map_in2=E_nilc, lmax_sht=lmax, beam=beam_0[:,0], beam2=beam_0[:,1], reuse_wsp=Xwsp_nmt)

    # if sim == start_sim:
    #     Dl_BB_dr, Bwsp_nmt = cf.map2Cl_nmt(B_dr, mask_apo, bins_BB, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, return_wsp=True)
    # else:
    #     Dl_BB_dr = cf.map2Cl_nmt(B_dr, mask_apo, bins_BB, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    # resNl_BB = cf.map2Cl_nmt(B_cilc_noise, mask_apo, bins_BB, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    # resFGpD_BB = cf.map2Cl_nmt(B_res - B_cilc_noise, mask_apo, bins_BB, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    # resFG_BB = cf.map2Cl_nmt(B_res - B_cilc_noise - B_dr, mask_apo, bins_BB, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)

    # resDl_TT = cf.map2Cl_nmt(T_res, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,0],  reuse_wsp=Twsp_nmt)
    # resDl_EE = cf.map2Cl_nmt(E_res, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,1],  reuse_wsp=Ewsp_nmt)
    # resDl_BB = cf.map2Cl_nmt(B_res, mask_apo, bins, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    # resDl_TE = cf.map2Cl_nmt(T_res, mask_c2, bins, map_in2=E_res * mask_20, lmax_sht=lmax, beam=beam_0[:,0], beam2=beam_0[:,1], reuse_wsp=Xwsp_nmt)

    resFGpD_TT = cf.map2Cl_nmt(T_res - T_nilc_noise, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,0],  reuse_wsp=Twsp_nmt)
    resFGpD_EE = cf.map2Cl_nmt(E_res - E_nilc_noise, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,1],  reuse_wsp=Ewsp_nmt)
    resFGpD_BB = cf.map2Cl_nmt(B_res - B_cilc_noise, mask_apo, bins, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    resFGpD_TE = cf.map2Cl_nmt(T_res - T_nilc_noise, mask_c2, bins, map_in2=(E_res - E_nilc_noise) * mask_20, lmax_sht=lmax, beam=beam_0[:,0], beam2=beam_0[:,1], reuse_wsp=Xwsp_nmt)
    
    resNl_TT = cf.map2Cl_nmt(T_nilc_noise, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,0],  reuse_wsp=Twsp_nmt)
    resNl_EE = cf.map2Cl_nmt(E_nilc_noise, mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,1],  reuse_wsp=Ewsp_nmt)
    resNl_BB = cf.map2Cl_nmt(B_cilc_noise, mask_apo, bins, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    # del T_nilc, E_nilc, B_cilc, T_nilc_noise, E_nilc_noise, B_cilc_noise

    # Dl_TT_cmb = cf.map2Cl_nmt(TEmap_act[0], mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,0], reuse_wsp=Twsp_nmt)
    # Dl_EE_cmb = cf.map2Cl_nmt(TEmap_act[1], mask_c2, bins, lmax_sht=lmax, beam=beam_0[:,1], reuse_wsp=Ewsp_nmt)
    # Dl_BB_cmb = cf.map2Cl_nmt(Bmap_act, mask_apo, bins, lmax_sht=lmax, beam=beam_0[:,2], masked_on_input=False, reuse_wsp=Bwsp_nmt)
    # Dl_TE_cmb = cf.map2Cl_nmt(TEmap_act[0], mask_c2, bins, map_in2=TEmap_act[1], lmax_sht=lmax, beam=beam_0[:,0], beam2=beam_0[:,1], reuse_wsp=Xwsp_nmt)

    Dl_TT_cmb, Dl_EE_cmb, Dl_BB_cmb, Dl_TE_cmb = np.loadtxt(output_root+'/data/Dl_actcmb_sim'+str(sim).zfill(3)+'.dat')[[1,2,3,4]]

    np.savetxt(output_root+'/data/Dl_tot_sim'+str(sim).zfill(3)+'.dat', [leff, Dl_TT[0], Dl_EE[0], Dl_BB[0], Dl_TE[0], \
    resFGpD_TT[0], resFGpD_EE[0], resFGpD_BB[0], resFGpD_TE[0], \
    resNl_TT[0], resNl_EE[0], resNl_BB[0]])

    np.savetxt(output_root+'/data/Dl_actcmb_sim'+str(sim).zfill(3)+'.dat', [leff, Dl_TT_cmb, Dl_EE_cmb, Dl_BB_cmb, Dl_TE_cmb])

    # np.savetxt(output_root+'/data/no_fix_bin/Dl_BB_deproj_res_sim'+str(sim).zfill(3)+'.dat', [leff_BB, Dl_BB_dr[0], resNl_BB[0], resFGpD_BB[0], resFG_BB[0]])
    # tot_Dl = np.loadtxt(output_root+'/data/no_fix_bin/Dl_BB_deproj_res_sim'+str(sim).zfill(3)+'.dat')
    # Dl_BB_dr, resNl_BB, resFGpD_BB, resFG_BB = tot_Dl[[1,2,3,4]]

    # tot_Dl = np.loadtxt(output_root+'/data/no_fix_bin/Dl_tot_sim'+str(sim).zfill(3)+'.dat')
    # Dl_BB, Dl_BB_cmb = tot_Dl[[3, 14]]

    # if sim == 0:
    #         Dl_TT_mean, Dl_EE_mean, Dl_BB_mean, Dl_TE_mean, \
    #         resDl_TT_mean, resDl_EE_mean, resDl_BB_mean, resDl_TE_mean, \
    #         resNl_TT_mean, resNl_EE_mean, resNl_BB_mean = \
    #         [], [], [], [], \
    #         [], [], [], [], \
    #         [], [], []
    # leff, Dl_TT, Dl_EE, Dl_BB, Dl_TE, \
    # resDl_TT, resDl_EE, resDl_BB, resDl_TE, \
    # resNl_TT, resNl_EE, resNl_BB, \
    # Dl_TT_cmb, Dl_EE_cmb, Dl_BB_cmb, Dl_TE_cmb = np.loadtxt(output_root+'/data/Dl_tot_sim'+str(sim).zfill(3)+'.dat')
    # Dl_TT_mean.append(Dl_TT)
    # Dl_EE_mean.append(Dl_EE)
    # Dl_BB_mean.append(Dl_BB)
    # Dl_TE_mean.append(Dl_TE)
    # resDl_TT_mean.append(resDl_TT)
    # resDl_EE_mean.append(resDl_EE)
    # resDl_BB_mean.append(resDl_BB)
    # resDl_TE_mean.append(resDl_TE)
    # resNl_TT_mean.append(resNl_TT)
    # resNl_EE_mean.append(resNl_EE)
    # resNl_BB_mean.append(resNl_BB)


    fno, fig, ax = cf.make_plotaxes()
    ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_tt[40:lmax_o+1], 'k-', label='TT input')
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_cmb[0, 40:lmax_o+1] / beam_0[40:lmax_o+1,0]**2., '-', label='TT anafast')
    ax.plot(leff[1:], Dl_TT[0,1:], 'o', lw=1.8, alpha=0.7, ms=3., label='NILC')
    ax.plot(leff[1:], resNl_TT[0,1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual noise')
    ax.plot(leff[1:], resFGpD_TT[0,1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual - res.noise')
    # ax.plot(leff[1:], Dl_TT_cmb[0,1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{TT}_\ell$ [in $\mu$K${}^2$]')
    ax.legend(loc='best', frameon=False, fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e0, ymax=1.e4)
    # ax.grid(which='both', axis='both')
    plt.savefig(output_root+'/plots/DlTT_NILC_sim_'+str(sim)+'.png',bbox_inches='tight',pad_inches=0.1)

    fno, fig, ax = cf.make_plotaxes()
    ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_ee[40:lmax_o+1], 'k-', label='EE input')
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_cmb[1, 40:lmax_o+1] / beam_0[40:lmax_o+1,1]**2., '-', label='EE anafast')
    ax.plot(leff[1:], Dl_EE[0,1:], 'o', lw=1.8, alpha=0.7, ms=3., label='NILC')
    ax.plot(leff[1:], resNl_EE[0,1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual noise')
    ax.plot(leff[1:], resFGpD_EE[0,1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual - res.noise')
    # ax.plot(leff[1:], Dl_EE_cmb[0,1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{EE}_\ell$ [in $\mu$K${}^2$]')
    ax.legend(loc='best', frameon=False, fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e-3, ymax=1.e2)
    # ax.grid(which='both', axis='both')
    plt.savefig(output_root+'/plots/DlEE_NILC_sim_'+str(sim)+'.png',bbox_inches='tight',pad_inches=0.1)

    fno, fig, ax = cf.make_plotaxes()
    ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*(cl_te)[40:lmax_o+1], 'k-', label='TE input')
    ax.plot(leff[1:], (Dl_TE)[0, 1:], 'o', lw=1.8, alpha=0.7, ms=3., label='NILC')
    ax.plot(leff[1:], resFGpD_TE[0, 1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual')
    # ax.plot(leff[1:], Dl_TE_cmb[0,1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{TE}_\ell$ [in $\mu$K${}^2$]')
    ax.legend(loc='best', frameon=False, fontsize=12)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim(ymin=5.e-2, ymax=2.e2)
    # ax.grid(which='both', axis='both')
    plt.savefig(output_root+'/plots/DlTE_NILC_sim_'+str(sim)+'.png',bbox_inches='tight',pad_inches=0.1)

    fno, fig, ax = cf.make_plotaxes()
    ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_bb[40:lmax_o+1], 'k-', label='BB input')
    ax.plot(leff[1:], Dl_BB[0, 1:], 'o', lw=1.8, alpha=0.7, ms=3., label='cILC')
    ax.plot(leff[1:], resNl_BB[0, 1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual noise')
    ax.plot(leff[1:], resFGpD_BB[0, 1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual - res.noise')
    # ax.plot(leff[1:], Dl_BB_cmb[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
    ax.legend(loc='best', frameon=False, fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(ymin=1.e-3, ymax=1.e2)
    # ax.grid(which='both', axis='both')
    plt.savefig(output_root+'/plots/DlBB_cILC_sim_'+str(sim)+'.png',bbox_inches='tight',pad_inches=0.1)

    # fno, fig, ax = cf.make_plotaxes()
    # ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_bb[40:lmax_o+1], 'k-', label='BB input')
    # ax.plot(leff_BB[1:], Dl_BB[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='cILC')
    # ax.plot(leff_BB[1:], resNl_BB[1:], '-', lw=1.8, alpha=0.7, ms=3., label='res. noise')
    # ax.plot(leff_BB[1:], resFGpD_BB[1:], '-', lw=1.8, alpha=0.7, ms=3., label='total res. - res.noise')
    # ax.plot(leff_BB[1:], Dl_BB_dr[1:], '-', lw=1.8, alpha=0.7, ms=3., label='deproj. res.')
    # ax.plot(leff_BB[1:], resFG_BB[1:], '-', lw=1.8, alpha=0.7, ms=3., label='total res. - res.noise - deproj. res.')
    # ax.plot(leff_BB[1:], Dl_BB_cmb[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
    # ax.set_xlabel(r'$\ell$')
    # ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
    # ax.legend(loc='best', frameon=False, fontsize=12)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # ax.set_ylim(ymin=1.e-3, ymax=1.e2)
    # # ax.grid(which='both', axis='both')
    # plt.savefig(output_root+'/plots/DlBB_cILC_sim_'+str(sim)+'_nofixbin.png',bbox_inches='tight',pad_inches=0.1)

    # del Dl_TT, Dl_EE, Dl_TE, Dl_BB, TEmap_act, Bmap_act

# e_Dl_TT = np.std(Dl_TT_mean, axis=0)
# e_Dl_EE = np.std(Dl_EE_mean, axis=0)
# e_Dl_TE = np.std(Dl_TE_mean, axis=0)
# e_Dl_BB = np.std(Dl_BB_mean, axis=0)
# Dl_TT_mean = np.mean(Dl_TT_mean, axis=0)
# Dl_EE_mean = np.mean(Dl_EE_mean, axis=0)
# Dl_BB_mean = np.mean(Dl_BB_mean, axis=0)
# Dl_TE_mean = np.mean(Dl_TE_mean, axis=0)
# resDl_TT_mean = np.mean(resDl_TT_mean, axis=0)
# resDl_EE_mean = np.mean(resDl_EE_mean, axis=0)
# resDl_BB_mean = np.mean(resDl_BB_mean, axis=0)
# resDl_TE_mean = np.mean(resDl_TE_mean, axis=0)
# resNl_TT_mean = np.mean(resNl_TT_mean, axis=0)
# resNl_EE_mean = np.mean(resNl_EE_mean, axis=0)
# resNl_BB_mean = np.mean(resNl_BB_mean, axis=0)
    
# fno, fig, ax = cf.make_plotaxes()
# ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_tt[40:lmax_o+1], 'k-', label='TT input')
# ax.errorbar(leff[1:], Dl_TT_mean[1:], yerr=np.abs(e_Dl_TT[1:]),fmt='s', lw=1.8, alpha=0.7, ms=4., label='NILC')
# ax.plot(leff[1:], resNl_TT_mean[1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual noise')
# ax.plot(leff[1:], resDl_TT_mean[1:] - resNl_TT_mean[1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual - res.noise')
# # ax.plot(leff[1:], Dl_TT_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
# ax.set_xlabel(r'$\ell$')
# ax.set_ylabel(r'$\mathcal{D}^{TT}_\ell$ [in $\mu$K${}^2$]')
# ax.legend(loc='best', frameon=False, fontsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_ylim(ymin=1.e0, ymax=1.e4)
# # ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlTT_NILC.png',bbox_inches='tight',pad_inches=0.1)

# fno, fig, ax = cf.make_plotaxes()
# ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_ee[40:lmax_o+1], 'k-', label='EE input')
# ax.errorbar(leff[1:], Dl_EE_mean[1:], yerr=np.abs(e_Dl_EE[1:]),fmt='s', lw=1.8, alpha=0.7, ms=4., label='NILC')
# ax.plot(leff[1:], resNl_EE_mean[1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual noise')
# ax.plot(leff[1:], resDl_EE_mean[1:] - resNl_EE_mean[1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual - res.noise')
# # ax.plot(leff[1:], Dl_EE_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
# ax.set_xlabel(r'$\ell$')
# ax.set_ylabel(r'$\mathcal{D}^{EE}_\ell$ [in $\mu$K${}^2$]')
# ax.legend(loc='best', frameon=False, fontsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_ylim(ymin=1.e-3, ymax=1.e2)
# # ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlEE_NILC.png',bbox_inches='tight',pad_inches=0.1)

# fno, fig, ax = cf.make_plotaxes()
# ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*(cl_te)[40:lmax_o+1], 'k-', label='TE input')
# ax.errorbar(leff[1:], Dl_TE_mean[1:], yerr=np.abs(e_Dl_TE[1:]),fmt='s', lw=1.8, alpha=0.7, ms=4., label='NILC')
# ax.plot(leff[1:], (resDl_TE_mean)[1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual')
# # ax.plot(leff[1:], Dl_TE_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
# ax.set_xlabel(r'$\ell$')
# ax.set_ylabel(r'$\mathcal{D}^{TE}_\ell$ [in $\mu$K${}^2$]')
# ax.legend(loc='best', frameon=False, fontsize=12)
# ax.set_xscale('log')
# # ax.set_yscale('log')
# # ax.set_ylim(ymin=5.e-2, ymax=2.e2)
# # ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlTE_NILC.png',bbox_inches='tight',pad_inches=0.1)

# fno, fig, ax = cf.make_plotaxes()
# ax.plot(ells[40:lmax_o+1], Dell_factor[40:lmax_o+1]*cl_bb[40:lmax_o+1], 'k-', label='BB input')
# ax.errorbar(leff[1:], Dl_BB_mean[1:], yerr=np.abs(e_Dl_BB[1:]),fmt='s', lw=1.8, alpha=0.7, ms=4., label='cILC')
# ax.plot(leff[1:], resNl_BB_mean[1:], '-', lw=1.8, alpha=0.7, ms=3., label='residual noise')
# ax.plot(leff[1:], resDl_BB_mean[1:] - resNl_BB_mean[1:], '-', lw=1.8, alpha=0.7, ms=3., label='total residual - res.noise')
# # ax.plot(leff[1:], Dl_BB_cmb_mean[1:], 'o', lw=1.8, alpha=0.7, ms=3., label='Actual CMB')
# ax.set_xlabel(r'$\ell$')
# ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
# ax.legend(loc='best', frameon=False, fontsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# # ax.set_ylim(ymin=1.e-3, ymax=1.e2)
# # ax.grid(which='both', axis='both')
# plt.savefig(output_root+'/DlBB_cILC.png',bbox_inches='tight',pad_inches=0.1)