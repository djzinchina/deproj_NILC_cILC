## deproj_NILC_cILC

This project includes codes for NILC/cILC on deprojected simulations. We aim to implement component separation methods on the input multifrequency maps whose systematics are filtered by deprojection. See [arXiv:2412.20415](https://arxiv.org/abs/2412.20415) for details of the work.

#### Main scripts

- `AliCPT_lens_fgcleaned-beamsys.py` uses NILC and cILC methods to implement foreground cleaning on AliCPT simulations with beam systematics. This is our main pipeline.
- `likelihood_only_r.py` samples the posterior distribution of $r$ from cleaned BB power spectrum.
- `synfast_noise.py` produces input noise simulations.
- `read_deproj_params.ipynb` checks if the recovered deprojection coefficients fit well with the input ones.

#### Plot folder

- `plot_95_150.py` plots the frequency maps at Ali 95 and 150 GHz.
- `plot_diff_fg.py` plots the differential foreground maps between Ali 95 GHz and Planck 100 GHz.
- `plot_Dl_map.py` plots the foreground-cleaned maps and their TT/EE/TE/BB power spectra.
- `plot_msk.py` plots the masks we used.

#### FG_clean folder

- `chilc.py` the constrained ILC method, used to clean B modes.
- `nilc_weights2.py/covarifast.f90/super_pix2.py` the NILC method, used to clean T and E modes.
- `ngls.py` the Needlet Generalized Least Square method, an alternative to NILC.

Thank you.

