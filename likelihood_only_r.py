import numpy as np 
import healpy as hp
import cmbframe as cf
import emcee as mc
import pygtc as gtc
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.random import default_rng
import astropy.io.fits as fits
from tqdm import tqdm 
import scipy


nside = 1024
lmax = 1500
lmax_o = 1000
nsims = 100
start_sim = 0

nwalkers = 48

r_fid = 0.0
bin_size = 30

bin_i = 1
bin_f = 5

output_root = '/home/doujzh/Documents/AliCPT_beamsys/output/'
resources_root = '/media/doujzh/AliCPT_data/AliCPT_lens2/data/'

msk_aff = 'UNPinvNvar'
cases = ['_lens_deproj']
for case in cases:
    method = ['_cnilc', '_cilc'][1]
    case_aff = msk_aff + method + case
    print("case :", case_aff)
    r = 0.0


    # msk_cut_aff = 'UNPinvNvar_newN0'

    plt.style.use('seaborn-ticks')
    plt.rc('font', family='sans-serif', size=12)
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    my_dpi = 600

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

    cl_tt, cl_ee, cl_bb_0p023, cl_te, cl_pp, cl_tp, cl_ep = import_PSM_spec()

    cl_bb_lens = import_PSM_spec(file_type='lensed')[2] - cl_bb_0p023
    cl_bb_fid = cl_bb_lens + cl_bb_0p023 * r_fid/0.023

    bins, leff, bsz, lmins = cf.setup_bins(nside, lmax_o)
    Cl_fid = cf.binner(cl_bb_fid, lmax, bsz, leff, is_Cell = False)[bin_i:bin_f+1]

    def ClBB_theory(r, A_lens=1.):
        Cl_th = cl_bb_0p023[0:lmax+1] * (r / 0.023) + (A_lens * cl_bb_lens[0:lmax+1])
        return cf.binner(Cl_th, lmax, bsz, leff, is_Cell = False)[bin_i:bin_f+1]

    def g_func(x):
        return np.sign(x - 1.) * np.sqrt(2. * (x - np.log(np.abs(x)) - 1.))

    Dl_act_mean= []
    Cl_nodb = []
    for sim in tqdm(range(start_sim, nsims), ncols=120):
        Dl_arr = np.loadtxt(resources_root+'Dl_tot_sim'+str(sim).zfill(3)+'.dat')
#       leff, Dl_TT[0], Dl_EE[0], Dl_BB[0], Dl_TE[0], \
#       resDl_TT[0], resDl_EE[0], resDl_BB[0], resDl_TE[0], \
#       resNl_TT[0], resNl_EE[0], resNl_BB[0], \
#       Dl_TT_cmb[0], Dl_EE_cmb[0], Dl_BB_cmb[0], Dl_TE_cmb[0]
        Dl_BB_act = Dl_arr[3]
        Dl_act_mean.append(Dl_BB_act)
        if sim == start_sim:
            ls = Dl_arr[0]
            Dell_factor = ls * (ls + 1.) / 2. / np.pi
        Dl_nodb = Dl_arr[3] + Dl_arr[11]
        Cl_nodb.append(Dl_nodb / Dell_factor)
    Cl_nodb = np.array(Cl_nodb)
    Dl_act_mean = np.mean(np.array(Dl_act_mean), axis=0)

    nbins = len(Cl_nodb[0,:])
    Mfid = np.zeros((nbins, nbins))
    for b1 in range(nbins):
        for b2 in range(nbins): 
            if b1 == b2:
                Mfid[b1,b2] = np.var(Cl_nodb[:,b1])
            if b1 != b2:
                Mfid[b1,b2] = Mfid[b2, b1] = np.cov(Cl_nodb[:,b1], Cl_nodb[:,b2])[0,1]
    Mfid = Mfid[bin_i:bin_f+1, bin_i:bin_f+1]
    # print(input_data.shape)
    Dl_hat = Dl_act_mean

    Cl_hat = (Dl_hat / Dell_factor)[bin_i:bin_f+1]

    Msize = len(Cl_hat)

    for i in range(Msize):
        for j in range(min(i+2, Msize), Msize):
            Mfid[i,j] = Mfid[j,i] = 0.


    det_Mfid = np.linalg.det(Mfid)
    Mfid_inv = np.linalg.inv(Mfid)

    def log_prior(r):
        # if -1 < r < 1:
        if 0. <= r < 1:
            return 0.
        else:
            return -np.inf

    def HL_likelihood(r, A_lens=1.):
        logP = log_prior(r)

        if np.isfinite(logP):
            Cl = ClBB_theory(r, A_lens=A_lens)
            g_Cl = np.reshape(g_func(Cl_hat / Cl) * Cl_fid,(len(Cl), 1))
            
            logL = -0.5 * np.matmul(g_Cl.T, np.matmul(Mfid_inv, g_Cl))

            if np.isnan(logL):
                print(r)
            return logP + logL 
        else:
            return logP

    def gaussian_likelihood(r, A_lens=1.):
        logP = log_prior(r)

        if np.isfinite(logP):
            Cl = ClBB_theory(r, A_lens=A_lens)
            d_Cl = Cl_hat - Cl
            
            logL = -0.5 * (np.matmul(d_Cl.T, np.matmul(Mfid_inv, d_Cl)) + np.log(det_Mfid))

            return logP + logL 
        else:
            return logP

    # mask_apo = hp.read_map('/media/doujzh/AliCPT_data/NoiseVar_MERRA_2/40Hz/AliCPT_UNPf_invNvar.fits', field=0, dtype=np.float64)
    def chi_sq(r, A_lens=1.):
        Cl = ClBB_theory(r, A_lens=A_lens)

        # b, leff, bin_sz, ell_min = cf.setup_bins(nside, lmax_o, is_Dell=False)
        # bin_sz = bin_sz[bin_i:bin_f+1]
        # fsky_apo = (hp.nside2pixarea(nside) / 4 * np.pi) * np.sum(mask_apo**2.)**2. / np.sum(mask_apo**4.)
        # chi2 = 0.
        # dof = 0.
        # for bin_no in range(bin_f + 1 - bin_i):
        #     LL =  bin_i + bin_no + 1
        #     chi2 += (2. * LL + 1.) * fsky_apo * bin_sz[bin_no] * ( Cl_hat[bin_no]/Cl[bin_no] - 1. - np.log(np.abs(Cl_hat[bin_no]/Cl[bin_no])))
        #     dof += (2. * LL + 1.) * fsky_apo * bin_sz[bin_no]
        
        # dof -= 1
        # print(chi2, dof)

        del_Cl = Cl_hat - Cl 
        del_Cl = np.reshape(del_Cl, (1, len(del_Cl)))
        chi2 = del_Cl @ Mfid_inv @ del_Cl.T
        dof = bin_f + 1 - bin_i - 1

        return chi2 / dof


    rng = default_rng()

    # position = np.reshape(rng.uniform(-0.05,0.05,nwalkers),(nwalkers, 1))
    position = np.reshape(rng.uniform(0.,0.05,nwalkers),(nwalkers, 1))
    ndim = 1

    case_aff += '_'+str(bin_i)+'-'+str(bin_f)
    # HLsampler = mc.EnsembleSampler(nwalkers, ndim, HL_likelihood)
    # HLsampler.run_mcmc(position, 10000, progress=True)

    GLsampler = mc.EnsembleSampler(nwalkers, ndim, gaussian_likelihood)
    GLsampler.run_mcmc(position, 10000, progress=True)

    # HL_chain = HLsampler.get_chain()
    GL_chain = GLsampler.get_chain()

    # fig, ax = plt.subplots(figsize=(8., 3.), dpi=my_dpi)
    # ax.plot(HL_chain[:,:,0], alpha=0.6, lw=0.3)
    # ax.set_xlim(0, len(HL_chain))
    # ax.set_xlabel('Step number')
    # ax.set_ylabel('r')
    # plt.savefig(output_root+'HL_MC-chain.png', bbox_inches='tight', pad_inches=0.1)

    # fig, ax = plt.subplots(figsize=(8., 3.), dpi=my_dpi)
    # ax.plot(GL_chain[:,:,0], alpha=0.6, lw=0.3)
    # ax.set_xlim(0, len(GL_chain))
    # ax.set_xlabel('Step number')
    # ax.set_ylabel('r')
    # plt.savefig(output_root+'r_GL_MC-chain.png', bbox_inches='tight', pad_inches=0.1)

    # flat_HLsamples = HLsampler.get_chain(discard=100, thin=15, flat=True)
    # # print(flat_HLsamples.shape)

    flat_GLsamples = GLsampler.get_chain(discard=100, thin=10, flat=True)
    # print("Shape", flat_GLsamples.shape)
    # np.savez(output_root+'data/r-MC_samples_'+case_aff+'.npz', cNILC_r=flat_GLsamples[:,0])
    theta_MAP = flat_GLsamples[np.argmax(GLsampler.get_log_prob(flat=True,discard=300, thin=10))]
    theta_MMSE = np.mean(flat_GLsamples, axis=0)
    # theta_MAP = np.median(flat_GLsamples, axis=0)

    print("Max. a-posteriori: ", theta_MAP, "Chi-sq/dof = ", chi_sq(theta_MAP[0]))
    print("Min. mean sq. error: ", theta_MMSE, "Chi-sq/dof = ", chi_sq(theta_MMSE[0]))

    r_1sigma = np.percentile(flat_GLsamples[:,0], [16, 50, 84])
    r_2sigma = np.percentile(flat_GLsamples[:,0], [5, 95])

    print("1 and 2 sigma: ", np.percentile(flat_GLsamples[:,0], [68,95]))
    print("r = ", theta_MMSE[0], '-', theta_MMSE[0] - r_1sigma[0], '+', r_1sigma[2] - theta_MMSE[0])
    if r>0:
        np.savetxt(output_root+'r-only_likelihood_cNILC_'+case_aff+'.dat', [theta_MMSE[0], theta_MMSE[0] - r_1sigma[0], r_1sigma[2] - theta_MMSE[0],
        theta_MMSE[0] - r_2sigma[0], r_2sigma[1] - theta_MMSE[0]])
        # 1-sigma-CI(mean, minus, plus), 2-sigma-CI(minu, plus)
    else:
        np.savetxt(output_root+'r-only_likelihood_cNILC_'+case_aff+'.dat', [theta_MMSE[0], np.percentile(flat_GLsamples[:,0], [95])])
        # mean, 95%-up-limit  

    truth1d = [
                [r], 
                # [np.mean(flat_HLsamples[:,0])], 
                theta_MMSE
                # [np.percentile(flat_GLsamples[:,0], 68)],
                # [np.percentile(flat_GLsamples[:,0], 95)]
                ]
    def plot_hist(data, ax, label, color):
        hist, edges = np.histogram(data, bins=30, density=True)
        centers = (edges[1:] + edges[:-1])/2
        x, y = scipy.ndimage.gaussian_filter1d((centers, hist), sigma=1)
        ax.plot(x, y/np.max(y), lw=1., alpha=1., color=color, label=label)
    # ax.plot(0,0,color='white', label=label)


    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=600)
    plt.style.use('seaborn-ticks')
    plt.rc('font', family='sans-serif', size=6)
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plot_hist(flat_GLsamples[:,0], ax, 'cILC', 'C0')
    ax.axvline(x=np.percentile(flat_GLsamples[:,0], [95]), c='k', lw='0.7', ls='--', label='95% CL.')
    ax.set_xlabel(r'$r$', fontsize=7)
    ax.set_ylabel(r'$P/P_{\rm max}$', fontsize=7)
    labels = ax.legend(loc='best', frameon=False, fontsize=6.5, ncol=1).get_texts()
    labels[0].set_color("C0")
    ax.set_xlim(xmin=0, xmax=0.1)
    ax.set_ylim(ymin=0, ymax=1)
    plt.savefig(output_root+'r-only_likelihood_'+case_aff+'.png',bbox_inches='tight')

# plt.show()
