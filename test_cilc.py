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
map_fwhm = 52.8 # arcmin
# overwrite_inp = True
constrain = [1,2]
nsims = 100
start_sim = 0
msk_aff = 'UNPinvNvar'
case_aff = msk_aff + '_cilc_deproj_beam52.8'
print("Case: ", case_aff)

cmb_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/BeamSys/CMB/'
lens_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/LensWithBeamSys/'
output_root = '/home/doujzh/Documents/AliCPT_beamsys/output/'
noise_root = '/media/doujzh/Ancillary_data/cNILC_covariance_matrix/resources/Noise_sims/'
data_root = '/media/doujzh/AliCPT_data2/Zirui_beamsys/data/'

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
fsky = np.sum(msk**2.) / hp.nside2npix(nside)
fsky_apo = np.sum(msk_apo**2.) / hp.nside2npix(nside)

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
# ================= start iteration ==============================
print("Iterating cILC cleaned maps")

Dl_nodb_mean = []
Nl_bias_mean = []
Nl_act_mean = []
Dl_act_mean = []
Dl_fg_mean = []
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
        if sim == 0 and nu == 4: cf.plot_maps(Bnoise*msk_20c2, mask_in=msk_20, proj='orth', outfile=output_root+'/maps/Bnoise_150GHz.png', show=False)
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

    # Set instr to point to WMAP, HFI 100-353 AliCPT 95-150 in that order
    Bcleaner = chilc.cilc_cleaner(Blms, beams=beams[:,:,2], com_res_beam=beam_0[:,2])
    Bcleaner.compute_cilc_weights(map_sel, constr_comp=constrain, bandpass='real', lmax_ch=lmax_ch)
    ilc_wgts = np.reshape(Bcleaner.har_wgts.T, (lmax+1, nu_dim,1))    # shape = (lmax+1, nu_dim, 1)
    np.savetxt(data_root+'cILC_wgts_'+case_aff+'_'+str(sim).zfill(3)+'.dat', Bcleaner.har_wgts)

    ilc_Blm = Bcleaner.get_cleaned_alms()
    ilc_Nlm = Bcleaner.get_projected_alms(Nlms)
    ilc_fglm = Bcleaner.get_projected_alms(fglms)
    if sim == start_sim: cf.plot_ilc_weights(Bcleaner.har_wgts, '', label_list=freqs, outfile=output_root+'cilc_wgts_'+case_aff+'.png', show=False)

    if sim == start_sim:
        Dl_fg, B_wsp = compute_Dell(ilc_fglm, return_wsp=True)
    else:
        Dl_fg = compute_Dell(ilc_fglm, reuse_wsp=B_wsp)
    Dl_fg_mean.append(Dl_fg)

    Nl_coup_BB = map2coup(ilc_Nlm)
    Nl_BB_act = compute_Dell(ilc_Nlm, reuse_wsp=B_wsp)

    Cl_coup_BB = map2coup(ilc_Blm)
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

# ####################### NOISE DEBIASING ################################
# Dl_arr = np.loadtxt(output_root+'Dl-mean_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.dat')
# leff, Dl_act_mean, Dl_fg_mean, Dl_nodb_mean, Nl_act_mean, Dl_act_err = Dl_arr
# print("Reading Nl data.")
# Nls_B = []
# for sim in tqdm(range(start_sim, start_sim+nsims), ncols=120):
#     Bnoise_path = noise_root + str(sim) + '/'
#     # Path(Bnoise_path).mkdir(parents=True, exist_ok=True)
#     Nlms = []
#     for nu in range(nu_dim):
#         Bnoise = hp.read_map(Bnoise_path + 'Bnoise_'+ freqs[nu]+'.fits', field=0, dtype=np.float64)
#         Nlms.append(hp.map2alm(Bnoise * msk_apo, lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.80/data'))
#         del Bnoise
#     Nlms = np.array(Nlms)

#     Nl_sim = np.zeros((lmax+1, nu_dim, nu_dim))
#     for i in range(nu_dim):
#         for j in range(i, nu_dim):
#             Nl_sim[:,i,j] = Nl_sim[:,j,i] = hp.alm2cl(np.copy(Nlms[i]), alms2=np.copy(Nlms[j]), lmax=lmax) * (beam_0[:,2]** 2. / beams[i,:,2] / beams[j,:,2])
#             # if np.any(np.isnan(Nl_sim[:,i,j])): print(i,j, np.where(np.isnan(Nl_sim[:,i,j])))
#     Nls_B.append(Nl_sim)
# Nls_B = np.array(Nls_B) # (nsims, lmax+1, nu_dim, nu_dim)
# np.save(noise_root+'Ncov_'+msk_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.npy', Nls_B)
# print(Nls_B.shape)
print("Computing noise bias.")
Nls_B = np.load(noise_root+'Ncov_'+msk_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.npy')
Dl_db_mean, Nl_db_mean = [], []
for sim in tqdm(range(start_sim, start_sim+nsims), ncols=120):
    slc = np.delete(np.arange(start_sim, start_sim+nsims), sim)
    Nl_slice = Nls_B[slc] # (nsims-1, lmax+1, nu_dim, nu_dim)
    har_wgts = np.loadtxt(data_root+'cILC_wgts_'+case_aff+'_'+str(sim).zfill(3)+'.dat')
    ilc_wgts = np.reshape(har_wgts.T, (lmax+1, nu_dim,1))
    Nl_proj = np.matmul(np.transpose(ilc_wgts, (0,2,1)), np.matmul(Nl_slice, ilc_wgts))[:,:,0,0] # (nsims-1, lmax+1)
    del Nl_slice
    if sim == start_sim: Fl, B_wsp = compute_Dell(hp.map2alm(fg[0][0], lmax=lmax, pol=False, use_weights=True, datapath='/home/doujzh/Healpix_3.80/data'), return_wsp=True)
    Nl_coup_BB = np.mean(Nl_proj, axis=0) / beam_0[:,2]** 2
    Nl_coup_BB = np.reshape(Nl_coup_BB,(1,len(Nl_coup_BB)))
    Nl_BB_db = B_wsp.decouple_cell(Nl_coup_BB)[0]
    
    Dl_BB_nodb = np.loadtxt(data_root+'Dl_tot_'+case_aff+'_'+str(sim).zfill(3)+'.dat')[3]
    Dl_BB_db = Dl_BB_nodb - Nl_BB_db

    Dl_db_mean.append(Dl_BB_db)
    Nl_db_mean.append(Nl_BB_db)
    np.savetxt(data_root+'Dl_db_'+case_aff+'_'+str(sim).zfill(3)+'.dat', [Dl_BB_db, Nl_BB_db])
Nl_db_err = np.std(np.array(Nl_db_mean), axis=0)
Nl_db_mean = np.mean(np.array(Nl_db_mean), axis=0)
Dl_db_err = np.std(np.array(Dl_db_mean), axis=0)
Dl_db_mean = np.mean(np.array(Dl_db_mean), axis=0)

np.savetxt(output_root+'Dl_db-mean_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.dat', [Dl_db_mean, Nl_db_mean, Dl_db_err, Nl_db_err])
# Dl_db_mean, Nl_db_mean, Dl_db_err, Nl_db_err = np.loadtxt(output_root+'/data/Dl_db-mean_r-'+str(r)+'_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.dat')

fno, fig, ax = cf.make_plotaxes() 
ax.plot(ell[30:], DlBB_lensed[30:], 'k-', lw=2, alpha=0.7, label='lensed CMB')
ax.plot(leff[1:], Nl_act_mean[1:], '--', lw=1.5, alpha=0.7, label='Actual Noise')
ax.plot(leff[1:], Nl_db_mean[1:], '--', lw=1.5, alpha=0.7, label='Noise Bias')
# ax.plot(leff[1:], Nl_db_err[1:], '--', lw=1.5, alpha=0.7, label='std of Nl')
ax.plot(leff[1:], Nl_db_mean[1:] - Nl_act_mean[1:], '-', lw=1.5, alpha=0.7, label='Nl_bias - Nl_act')
ax.errorbar(leff[1:], Dl_db_mean[1:], yerr=np.abs(Dl_db_err[1:]),fmt='s', lw=1.8, alpha=0.7, ms=4., label='est. Dl')
ax.plot(ell[30:], DlBB[30-2:lmax_o+1-2], 'k--', lw=2, alpha=0.7, label='r=0.023')
ax.plot(leff[1:], Dl_fg_mean[1:], '-', lw=1.5, alpha=0.7, label='Residual diff. FG')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{BB}_\ell$ [in $\mu$K${}^2$]')
ax.legend(loc='best', frameon=False, fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(ymin=1.e-7, ymax=1.e2)
# ax.grid(which='both', axis='both')
plt.savefig(output_root+'Test_cNILC_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.png',bbox_inches='tight',pad_inches=0.1)

# plt.show()
fno, fig, ax = cf.make_plotaxes()
# ax.plot(leff[1:], DlBB_tens[1:], 'o', lw=2, alpha=0.7, label='theo. bandpowers')
# ax.plot(ell[30:], DlBB[30-2:lmax_o+1-2], 'k-', lw=2, alpha=0.7, label='r='+str(r))
ax.errorbar(leff[1:], (Dl_db_mean[1:] - DlBB_lens[1:])/DlBB_lens[1:], yerr=np.abs(Dl_db_err[1:])/DlBB_lens[1:],fmt='s', lw=1.8, alpha=0.7, ms=4., label='est. Dl - theo. Dl')
ax.plot(leff[1:], Dl_fg_mean[1:]/DlBB_lens[1:], '-', lw=1.5, alpha=0.7, label='Residual diff. FG')
ax.plot(leff[1:], (Nl_act_mean[1:] - Nl_db_mean[1:])/DlBB_lens[1:], '-', lw=1.5, alpha=0.7, label='Nl_act - Nl_bias')
ax.plot(ell[30:], np.zeros(len(ell[30:])), 'k--', lw=1, alpha=0.7)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\mathcal{D}^{bias}_\ell/\mathcal{D}^{BB}_\ell$')
ax.legend(loc='best', frameon=False, fontsize=12)
ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylim(ymin=5.e-5, ymax=1.e2)
ax.set_ylim(ymin=-0.45, ymax=0.45)
# ax.grid(which='both', axis='both')
plt.savefig(output_root+'Test_flat_cNILC_'+case_aff+'_nsims-'+str(nsims)+'_bsz'+str(bin_size)+'.png',bbox_inches='tight',pad_inches=0.1)


# plt.show()







