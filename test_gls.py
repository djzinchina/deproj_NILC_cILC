import numpy as np 
import healpy as hp
import cmbframe as cf 
import pymaster as nmt
from tqdm import tqdm
import astropy.io.fits as fits 
import matplotlib.pyplot as plt
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
map_fwhm = 52.8 # arcmin
compute_Ncov = True
constrain = [1,2]
nsims = 100
start_sim = 0
msk_aff = 'UNPinvNvar'
case_aff = msk_aff + '_gls_deproj_beam52.8'
print("Case: ", case_aff)

cmb_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/CMB/'
lens_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/LensWithBeamSys/'
output_root = '/home/doujzh/Documents/AliCPT_beamsys/output/'
noise_root = '/media/doujzh/Ancillary_data/cNILC_covariance_matrix/resources/Noise_sims/'
data_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/data/'
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
msk3 = [msk_alicpt, msk_alicpt, msk_alicpt]
msk_20 = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_20uKcut150_C_1024.fits', field=None, dtype=np.float64, verbose=False)
msk_20c2 = nmt.mask_apodization(msk_20, 6.0, apotype='C2')
msk_unp = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPfg_filled_C_1024.fits',field=0, dtype=None, verbose=False)
msk_unpinv = hp.read_map('/home/doujzh/DATA/AliCPT/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64, verbose=False)

foreground_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/FG/0/'
# cmb_root = '/media/doujzh/AliCPT_data/LensedCMB'
instrs = ['WMAP', 'Ali', 'HFI', 'HFI', 'Ali','HFI', 'HFI']
freqs = ['K', '95', '100', '143', '150', '217', '353']
map_sel = np.arange(len(freqs))
bands_beam = [52.8, 19., 9.682, 7.303, 11., 5.021, 4.944]

nu_dim = len(map_sel)
fg = []
beams = []
for nu in map_sel:
    file_name = instrs[nu]+'_'+freqs[nu]+'.fits'
    file_path = os.path.join(foreground_root, file_name)

    fg.append(hp.read_map(file_path, field=None, dtype=np.float64)*msk_alicpt)

    beams.append(hp.gauss_beam(np.deg2rad(bands_beam[nu] / 60.), pol=True, lmax=lmax))

# exit()
beams = np.array(beams)
# beam_0 = hp.gauss_beam(np.deg2rad(map_fwhm / 60.), pol=True, lmax=lmax)

####################### Load Data ############################
def fetch_cmbpfg(sim, deproj=False):
    cmbpfg = []
    for nu in map_sel:
        file_name = instrs[nu]+'_'+freqs[nu]+'.fits'
        if deproj and 'Ali' in file_name: file_name = 'DEPROJ_' + file_name
        file_path = os.path.join(lens_root, str(sim).zfill(3), file_name)
        cmbpfg_nu = hp.read_map(file_path, field=None, dtype=np.float64)

        cmbpfg.append(cmbpfg_nu * msk_alicpt)

        del cmbpfg_nu

    return cmbpfg
def fetch_noise(sim):
    noise_path = noise_root+str(sim)+'/'
    noise_out = []
    for nu in map_sel:
        noise_filename = noise_path + 'Noise_IQU_'+ freqs[nu] +'.fits'
        noise_out.append(hp.read_map(noise_filename, field=None, dtype=np.float64, verbose=False)*msk_alicpt)
    # noise_out = np.array(noise_out)
    return noise_out
def fetch_Nlms(sim):
    Nlms = []
    for nu in map_sel:
        file_name = 'Bnoise_'+freqs[nu]+'.fits'
        file_path = os.path.join(noise_root, str(sim), file_name)
        Bnoise = hp.read_map(file_path, field=0, dtype=np.float64)

        Nlms.append(hp.map2alm(Bnoise * msk_apo, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.80/data'))

        del Bnoise

    return np.array(Nlms)
##################### Binning Scheme ##########################

# print(nu_dim)
bins, leff, bsz, lmins = cf.setup_bins(nside, lmax_o, loglims=[[1.+bin_size/100]], bsz_0=bin_size)
ell = np.arange(lmax_o+1)
Dell_factor = ell * (ell + 1.) / 2. / np.pi
DlBB = cl_bb[:lmax_o+1] * Dell_factor
DlBB_tens = cf.binner(DlBB, lmax_o, bsz, leff, is_Cell = False)
DlBB_lensed = cl_bb_lensed[:lmax_o+1] * Dell_factor
DlBB_lens = cf.binner(DlBB_lensed, lmax_o, bsz, leff, is_Cell = False)
################### Case Loop ###########################

if 'UNP' in msk_aff:
    msk = msk_unp
    if 'inv' in msk_aff:
        msk_apo = msk_unpinv
    elif 'apo' in msk_aff:
        msk_apo = nmt.mask_apodization(msk_unp, 6.0, apotype='C2')
# if 'Mask' in msk_aff:
#     msk = hp.read_map(mask_root+"Mask_test.fits", field=0, dtype=np.float64, verbose=False)
#     msk_apo = hp.read_map(mask_root+"Maskapo_test.fits", field=0, dtype=np.float64, verbose=False)
fsky = np.sum(msk**2.) / hp.nside2npix(nside)
fsky_apo = np.sum(msk_apo)**2./ np.sum(msk_apo**2.)/ hp.nside2npix(nside)
print("fsky", fsky, fsky_apo)

if 'beam11' in case_aff:
    lmax_ch = np.array([300, 1000, lmax, lmax, lmax, lmax, lmax])
    beam_0 = hp.gauss_beam(np.deg2rad(11. / 60.), lmax=lmax, pol=True)
    freq_0 = '150'
elif 'beam52.8' in case_aff:
    lmax_ch = np.array([lmax, lmax, lmax, lmax, lmax, lmax, lmax])
    beam_0 = hp.gauss_beam(np.deg2rad(52.8 / 60.), lmax=lmax, pol=True)
    freq_0 = 'K'

def map2coup(Blm, return_fld=False):
    Bmap = hp.alm2map(Blm, nside, lmax=lmax, pol=False)
    fld_B = nmt.NmtField(msk_apo, [Bmap], lmax_sht=lmax, masked_on_input=True,)
    Cl_coup_BB = nmt.compute_coupled_cell(fld_B, fld_B)/beam_0[:,2]**2
    if return_fld:
        return Cl_coup_BB, fld_B
    else:
        return Cl_coup_BB
def compute_Dell(Blm, reuse_wsp=None, return_wsp=False):
    if isinstance(reuse_wsp, type(nmt.NmtWorkspace())):
        Cl_coup_BB = map2coup(Blm)
        wsp = reuse_wsp
        Dl_BB_nmt = wsp.decouple_cell(Cl_coup_BB)[0]
    else:
        Cl_coup_BB, fld_B = map2coup(Blm, return_fld=True)
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(fld_B, fld_B, bins)
        Dl_BB_nmt = wsp.decouple_cell(Cl_coup_BB)[0]
    if return_wsp:
        return Dl_BB_nmt, wsp
    else:
        return Dl_BB_nmt
################ Compute Ncov ###################
if compute_Ncov :
    Ncov_data = []

    for sim in tqdm(range(nsims)):

        Nlms = fetch_Nlms(sim)

        Ncov_data.append(cf.compute_Ncov(Nlms,  beams=beams[:,:,2], com_res_beam=beams[0,:,2]))

        del Nlms

    Ncov_data = np.array(Ncov_data)

    Ncov = np.mean(Ncov_data, axis=0)
    Ncov_std = np.std(Ncov_data, axis=0)

    del Ncov_data

    np.savetxt(noise_root+'Ncov_mean_'+msk_aff+'_nsims-'+str(nsims)+'.dat', np.reshape(Ncov, (len(Ncov[:,0,0]), len(Ncov[0,:,0]) * len(Ncov[0,0,:]))))
    np.savetxt(noise_root+'Ncov_std_'+msk_aff+'_nsims-'+str(nsims)+'.dat', np.reshape(Ncov_std, (len(Ncov[:,0,0]), len(Ncov[0,:,0]) * len(Ncov[0,0,:]))))

else: 
    Ncov = np.loadtxt(noise_root+'Ncov_mean_'+msk_aff+'_nsims-'+str(nsims)+'.dat')
    # print(Ncov.shape)
    ncov_nu = int(np.sqrt(len(Ncov[0,:])))
    Ncov = np.reshape(Ncov,(len(Ncov[:,0]), ncov_nu, ncov_nu))
    Ncov = Ncov[:,map_sel]
    Ncov = Ncov[:,:,map_sel]
# ================= start iteration ==============================
print("Iterating GLS cleaned maps")

Dl_nodb_mean = []
Nl_bias_mean = []
Nl_act_mean = []
Dl_act_mean = []
Dl_fg_mean = []

Bcleaner = cf.gls_solver(Ncov, instr=map_sel, comp=[0,1,2], bandpass='real')
for sim in tqdm(range(start_sim, start_sim+nsims), ncols=120):
    # save_noise(sim)
    cmbpfg = fetch_cmbpfg(sim, deproj=True)
    noise = fetch_noise(sim)
    Blms, Nlms, fglms = [], [], []
    # betas = []
    for nu in range(nu_dim):
        obsn_nu = cmbpfg[nu] + noise[nu]

        Bobs_nu, b0, b1 = cf.get_cleanedBmap(obsn_nu, msk_20, lmax_sht=lmax_rot, return_fit=True)
        # betas.append(b1)
        del obsn_nu
        Bnoise = cf.get_cleanedBmap(noise[nu], msk_20, beta_1=b1)
        # hp.write_map(noise_root + str(sim) + '/Bnoise_' + freqs[nu]+'.fits', Bnoise, overwrite=True)
        # if sim == 0 and nu == 4: cf.plot_maps(Bnoise*msk_20c2, mask_in=msk_20, proj='orth', outfile=output_root+'/maps/Bnoise_150GHz.png', show=False)
        Bfg = cf.get_cleanedBmap(fg[nu], msk_20, beta_1=b1)
        Nlms.append(hp.map2alm(Bnoise * msk_apo, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.80/data'))
        fglms.append(hp.map2alm(Bfg * msk_apo, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.80/data'))
        Blms.append(hp.map2alm(Bobs_nu * msk_apo, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.80/data'))

        del Bobs_nu
    # betas = np.array(betas)
    # np.savetxt(Cl_db_root+'TC_Betas_r'+str(r)+'_'+case_aff+'_'+str(sim).zfill(3)+'.dat', betas)
    Blms = np.array(Blms)
    Nlms = np.array(Nlms)
    fglms = np.array(fglms)

    # if np.any(np.isnan(Nl_data)):
    #     for i in range(150):
    #         if np.any(np.isnan(Nl_data[i])):
    #             print(i)

    gls_Blm = Bcleaner.get_component(Blms, 0, beams=beams[:,:,2], com_res_beam=beams[0,:,2])
    gls_Nlm = Bcleaner.get_component(Nlms, 0, beams=beams[:,:,2], com_res_beam=beams[0,:,2])
    gls_fglm = Bcleaner.get_component(fglms, 0, beams=beams[:,:,2], com_res_beam=beams[0,:,2])
    if sim == start_sim: cf.plot_gls_weights(Bcleaner.GLS_W, 0, label_list=freqs, outfile=output_root+'gls_wgts_'+case_aff+'.png', show=False)

    if sim == start_sim:
        Dl_fg, B_wsp = compute_Dell(gls_fglm, return_wsp=True)
    else:
        Dl_fg = compute_Dell(gls_fglm, reuse_wsp=B_wsp)
    Dl_fg_mean.append(Dl_fg)

    Nl_coup_BB = map2coup(gls_Nlm)
    Nl_BB_act = compute_Dell(gls_Nlm, reuse_wsp=B_wsp)

    Cl_coup_BB = map2coup(gls_Blm)
    Dl_BB_act = B_wsp.decouple_cell(Cl_coup_BB, cl_noise=Nl_coup_BB)[0]

    # # if np.any(np.isnan(Dl_BB_db)):
    # #     print(np.sum(np.isnan(Nl_coup_BB)))
    Dl_act_mean.append(Dl_BB_act)
    del Nl_coup_BB

    Nl_act_mean.append(Nl_BB_act)

    Dl_BB_nodb = B_wsp.decouple_cell(Cl_coup_BB)[0]

    Dl_nodb_mean.append(Dl_BB_nodb)

    np.savetxt(data_root+'Dl_tot_'+case_aff+'_'+str(sim).zfill(3)+'.dat', [leff, Dl_BB_act, Dl_fg, Dl_BB_nodb, Nl_BB_act])


Dl_act_err = np.std(np.array(Dl_nodb_mean), axis=0)
Dl_nodb_mean = np.mean(np.array(Dl_nodb_mean), axis=0)
Nl_act_mean = np.mean(np.array(Nl_act_mean), axis=0)
Dl_act_mean = np.mean(np.array(Dl_act_mean), axis=0)
Dl_fg_mean = np.mean(np.array(Dl_fg_mean), axis=0)

np.savetxt(output_root+'Dl-mean_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.dat', [leff, Dl_act_mean,
Dl_fg_mean, Dl_nodb_mean, Nl_act_mean, Dl_act_err])

fno, fig, ax = cf.make_plotaxes() 
ax.plot(ell[30:], DlBB_lensed[30:], 'k-', lw=2, alpha=0.7, label='lensed CMB')
ax.plot(leff[1:], Nl_act_mean[1:], '--', lw=1.5, alpha=0.7, label='Actual Noise')
ax.plot(leff[1:], Nl_act_mean[1:], '--', lw=1.5, alpha=0.7, label='Noise Bias')
# ax.plot(leff[1:], Nl_db_err[1:], '--', lw=1.5, alpha=0.7, label='std of Nl')
ax.errorbar(leff[1:], Dl_act_mean[1:], yerr=np.abs(Dl_act_err[1:]),fmt='s', lw=1.8, alpha=0.7, ms=4., label='est. Dl')
ax.plot(ell[30:], DlBB[30-2:lmax_o+1-2], 'k--', lw=2, alpha=0.7, label='r=0.023')
ax.plot(leff[1:], Dl_fg_mean[1:], '-', lw=1.5, alpha=0.7, label='Residual diff. FG')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='best', frameon=False, fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(ymin=1.e-7, ymax=1.e2)
# ax.grid(which='both', axis='both')
plt.savefig(output_root+'Test_GLS_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.png',bbox_inches='tight',pad_inches=0.1)

# plt.show()
fno, fig, ax = cf.make_plotaxes()
# ax.plot(leff[1:], DlBB_tens[1:], 'o', lw=2, alpha=0.7, label='theo. bandpowers')
# ax.plot(ell[30:], DlBB[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='r='+str(r))
ax.errorbar(leff[1:], (Dl_act_mean[1:] - DlBB_lens[1:])/DlBB_lens[1:], yerr=np.abs(Dl_act_err[1:])/DlBB_lens[1:],fmt='s', lw=1.8, alpha=0.7, ms=4., label='est. Dl - theo. Dl')
ax.plot(leff[1:], Dl_fg_mean[1:]/DlBB_lens[1:], '-', lw=1.5, alpha=0.7, label='Residual diff. FG')
ax.plot(ell[30:], np.zeros(len(ell[30:])), 'k--', lw=1, alpha=0.7)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{bias}_\ell/\mathcal{D}^{BB}_\ell$')
ax.legend(loc='best', frameon=False, fontsize=12)
ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylim(ymin=5.e-5, ymax=1.e2)
ax.set_ylim(ymin=-0.45, ymax=0.45)
# ax.grid(which='both', axis='both')
plt.savefig(output_root+'Test_flat_GLS_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.png',bbox_inches='tight',pad_inches=0.1)


# plt.show()







