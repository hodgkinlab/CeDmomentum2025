"""
Cyton2 fit for AF's CD4 or CD8 T cells (HD, CVID and COE)
Modify the Cyton2 model to estimate the initial cell numbers, rather than fixed as a constant
Last edit: 17-April-2023
"""
import os, sys, time, datetime, copy, tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import multiprocessing as mp
import lmfit as lmf
mpl.use('Agg')  # switch plot backend to avoid fork errors for parallel processing
rng = np.random.RandomState(seed=34095307)
options = {'strings_to_formulas': False, 'strings_to_urls': False}  # Required to save fit results to excel file

from src.parse import parse_data
from src.utils import conf_iterval, lognorm_cdf, lognorm_pdf
from src.model import Cyton2Model

rc = {
	'figure.figsize': (9, 7),
	# 'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 12,
	# 'xtick.labelsize': 14, 'ytick.labelsize': 14,
	# 'axes.grid': True, 'axes.grid.axis': 'x', 'axes.grid.axis': 'y',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1,
	'errorbar.capsize': 2.5
}
sns.set(style='white', rc=rc)

DT = 0.5              # [Cyton Model] Time step
ITER_SEARCH = 100     # [Cyton Model] Number of initial search
MAX_NFEV = None       # [LMFIT] Maximum number of function evaluation
LM_FIT_KWS = {        # [LMFIT/SciPy] Key-word arguements pass to LMFIT minimizer for Levenberg-Marquardt algorithm
	# 'ftol': 1E-10,  # Relative error desired in the sum of squares. DEFAULT: 1.49012E-8
	# 'xtol': 1E-10,  # Relative error desired in the approximate solution. DEFAULT: 1.49012E-8
	# 'gtol': 0.0,    # Orthogonality desired between the function vector and the columns of the Jacobian. DEFAULT: 0.0
	'epsfcn': 1E-4    # A variable used in determining a suitable step length for the forward-difference approximation of the Jacobian (for Dfun=None). Normally the actual step length will be sqrt(epsfcn)*x If epsfcn is less than the machine precision, it is assumed that the relative errors are of the order of the machine precision. Default value is around 2E-16. As it turns out, the optimisation routine starts by making a very small move (functinal evaluation) and calculating the finite-difference Jacobian matrix to determine the direction to move. The default value is too small to detect sensitivity of 'm' parameter.
}

RGS = 95              # [BOOTSTRAP] alpha for confidence interval
ITER_BOOTS = 1000     # [BOOTSTRAP] Bootstrap samples


### MODEL: RESIDUAL
def residual(pars, x, data=None, model=None):
	vals = pars.valuesdict()
	mUns, sUns = vals['mUns'], vals['sUns']
	mDiv0, sDiv0 = vals['mDiv0'], vals['sDiv0']
	mDD, sDD = vals['mDD'], vals['sDD']
	mDie, sDie = vals['mDie'], vals['sDie']
	m, p = vals['m'], vals['p']

	pred = model.evaluate(mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p)

	return (data - pred)

### Bootstrap
def bootstrap(key, df, cond, icnd, hts, nreps, params, paramExcl):
	# define set collectors
	boots = {
		'mUns': [], 'sUns': [],  # NOTE - Maybe useful when we utilise unstimulated info.
		'mDiv0': [], 'sDiv0': [],
		'mDD': [], 'sDD': [],
		'mDie': [], 'sDie': [],
		'm': [], 'p': [], 'N0': []
	}
	pos = mp.current_process()._identity[0]-1  # fix position of the progress bar
	for _ in tqdm.trange(ITER_BOOTS, desc=f"Bootstrap", leave=False, position=2*pos+1):
		pars = copy.copy(params)

		# SHUFFLE DATASET
		cells = df['cgens']['rep'][icnd]  # n(g,t): number of cells in generation g at time t
		gens_boot, cells_boot = [], []
		for itpt, (data, ht) in enumerate(zip(cells, hts)):
			_cohort = []
			for _ in range(len(data)):
				rand_idx = rng.randint(0, len(data))
				cohorts = 0.
				for igen, cell_number_rep in enumerate(data[rand_idx]):
					cohorts += cell_number_rep * np.power(2., -float(igen))
					gens_boot.append(igen)
					cells_boot.append(cell_number_rep)
				_cohort.append(cohorts)
			if not itpt:
				avg_init_cells = np.mean(_cohort)
		gens_boot = np.asfarray(gens_boot)
		cells_boot = np.asfarray(cells_boot)

		model = Cyton2Model(hts, avg_init_cells, int(max(gens_boot)), DT, nreps)

		candidates = {'result': [], 'residual': []}
		for _ in tqdm.trange(ITER_SEARCH, desc=f"Cyton2 > {key} > {cond}", leave=False, position=2*pos+2):
			# Random initial guesses
			for par in pars:
				if par in paramExcl: pass  # ignore excluded parameters.
				else:
					par_min, par_max = pars[par].min, pars[par].max  # sample range
					pars[par].set(value=rng.uniform(low=par_min, high=par_max))

			try:  # to avoid errors when the algorithm fail
				mini = lmf.Minimizer(residual, pars, fcn_args=(gens_boot, cells_boot, model), **LM_FIT_KWS)
				res = mini.minimize(method='leastsq', max_nfev=MAX_NFEV)  # Levenberg-Marquardt algorithm

				candidates['result'].append(res)
				candidates['residual'].append(res.chisqr)
			except ValueError as ve:
				pass
		fit_results = pd.DataFrame(candidates)
		fit_results.sort_values('residual', ascending=True, inplace=True)  # find lowest RSS

		boot_best_fit = fit_results.iloc[0]['result'].params.valuesdict()
		for var in boot_best_fit:
			boots[var].append(boot_best_fit[var])
		boots['N0'].append(avg_init_cells)
	boots = pd.DataFrame(boots)
	return boots


### Cyton2 fit routine
def fit(inputs):
	key, df, reader, icnd = inputs  # unpack input data

	hts = reader.harvested_times[icnd]
	mgen = reader.generation_per_condition[icnd]
	condition = reader.condition_names[icnd]

	# ORGANISE DATA
	data = df['cgens']['rep'][icnd]  # n(g,t): number of cells in generation g at time t
	x_gens, y_cells = [], []
	all_reps, nreps = [len(l) for l in data], []

	for itpt, (datum, ht) in enumerate(zip(data, hts)):
		_cohorts = []
		for irep, rep in enumerate(datum):
			cohorts = 0.
			for igen, cell_number in enumerate(rep):
				cohorts += cell_number * np.power(2., -float(igen))
				x_gens.append(igen)
				y_cells.append(cell_number)
			_cohorts.append(cohorts)
		nreps.append(irep+1)
		if not itpt:
			N0 = np.mean(_cohorts)
	x_gens = np.asfarray(x_gens)
	y_cells = np.asfarray(y_cells)

	# DEFINE PARAMETER PROPERTIES
	pars = {
		'mUns': 100_000, 'sUns': 1E-10,  # Unstimulated death time (NOT USED HERE)
		'mDiv0': 30, 'sDiv0': 0.2,     # Time to first division
		'mDD': 60, 'sDD': 0.3,         # Time to division destiny
		'mDie': 80, 'sDie': 0.2,       # Time to death
		'm': 10, 'p': 1                # Subsequent division time & Proportion of activated cells
	}
	bounds = {
		'lb': {  # Lower bounds
			'mUns': 1E-2, 'sUns': 1E-2,
			'mDiv0': 1E-2, 'sDiv0': 1E-2,
			'mDD': 1E-2, 'sDD': 1E-2,
			'mDie': 1E-2, 'sDie': 1E-2,
			'm': 0, 'p': 0
		},
		'ub': {  # Upper bounds
			'mUns': 100_000, 'sUns': 2,
			'mDiv0': 500, 'sDiv0': 2,
			'mDD': 500, 'sDD': 2,
			'mDie': 500, 'sDie': 2,
			'm': 50, 'p': 1
		}
	}
	vary = {  # True = Subject to change; False = Lock parameter
		'mUns': False, 'sUns': False,
		'mDiv0': True, 'sDiv0': True,
		'mDD': True, 'sDD': True,
		'mDie': True, 'sDie': True,
		'm': True, 'p': False
	}
	params = lmf.Parameters()
	for par in pars:
		params.add(par, value=pars[par], min=bounds['lb'][par], max=bounds['ub'][par], vary=vary[par])
	paramExcl = [p for p in params if not params[p].vary]  # List of locked parameters

	# RUN MODEL FITTING
	model = Cyton2Model(hts, N0, mgen, DT, nreps)

	candidates = {'result': [], 'residual': []}
	pos = mp.current_process()._identity[0]-1
	for _ in tqdm.trange(ITER_SEARCH, desc=f"Cyton2 > {key} > {condition}", leave=False, position=2*pos+1):
		# Random initial guesses
		for par in params:
			if par in paramExcl: pass  # Ignore locked parameters
			else:
				par_min, par_max = params[par].min, params[par].max
				params[par].set(value=rng.uniform(low=par_min, high=par_max))

		try:
			mini = lmf.Minimizer(residual, params, fcn_args=(x_gens, y_cells, model), **LM_FIT_KWS)
			res = mini.minimize(method='leastsq', max_nfev=MAX_NFEV)

			candidates['result'].append(res)
			candidates['residual'].append(res.chisqr)
		except ValueError as ve:
			pass
	fit_results = pd.DataFrame(candidates)
	fit_results.sort_values('residual', ascending=True, inplace=True)  # Find lowest RSS

	best_fit = fit_results.iloc[0]['result'].params.valuesdict()
	mUns, sUns = best_fit['mUns'], best_fit['sUns']
	mDiv0, sDiv0 = best_fit['mDiv0'], best_fit['sDiv0']
	mDD, sDD = best_fit['mDD'], best_fit['sDD']
	mDie, sDie = best_fit['mDie'], best_fit['sDie']
	m, p = best_fit['m'], best_fit['p']
	
	# RUN BOOTSTRAP
	boots = bootstrap(key, df, condition, icnd, hts, nreps, params, paramExcl)

	# SETTINGS FOR MODEL EXTRAPOLATION
	t0, tf = 0, max(hts)+5
	times = np.linspace(t0, tf, num=int(tf/DT)+1)
	gens = np.array([i for i in range(mgen+1)])

	# Calculate bootstrap intervals
	unst_pdf_curves, unst_cdf_curves = [], []
	tdiv0_pdf_curves, tdiv0_cdf_curves = [], []
	tdd_pdf_curves, tdd_cdf_curves = [], []
	tdie_pdf_curves, tdie_cdf_curves = [], []

	b_ext_total_live_cells, b_ext_cells_per_gen = [], []
	b_hts_total_live_cells, b_hts_cells_per_gen = [], []

	conf = {
		'unst_pdf': [], 'unst_cdf': [], 'tdiv0_pdf': [], 'tdiv0_cdf': [], 'tdd_pdf': [], 'tdd_cdf': [], 'tdie_pdf': [], 'tdie_cdf': [],
		'ext_total_cohorts': [], 'ext_total_live_cells': [], 'ext_cells_per_gen': [], 'hts_total_live_cells': [], 'hts_cells_per_gen': [],
		'ext_mdn': [], 'ext_mdn_ex0': []
	}
	tmp_N0 = []
	for _, bsample in boots.iterrows():
		b_mUns, b_sUns, b_mDiv0, b_sDiv0, b_mDD, b_sDD, b_mDie, b_sDie, b_m, b_p, b_N0 = bsample.values
		b_params = params.copy()
		b_params['mUns'].set(value=b_mUns); b_params['sUns'].set(value=b_sUns)
		b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
		b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
		b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
		b_params['m'].set(value=b_m); b_params['p'].set(value=b_p)

		# Calculate PDF and CDF curves for each set of parameter
		b_unst_pdf, b_unst_cdf = lognorm_pdf(times, b_mUns, b_sUns), lognorm_cdf(times, b_mUns, b_sUns)
		b_tdiv0_pdf, b_tdiv0_cdf = lognorm_pdf(times, b_mDiv0, b_sDiv0), lognorm_cdf(times, b_mDiv0, b_sDiv0)
		b_tdd_pdf, b_tdd_cdf = lognorm_pdf(times, b_mDD, b_sDD), lognorm_cdf(times, b_mDD, b_sDD)
		b_tdie_pdf, b_tdie_cdf = lognorm_pdf(times, b_mDie, b_sDie), lognorm_cdf(times, b_mDie, b_sDie)

		unst_pdf_curves.append(b_unst_pdf); unst_cdf_curves.append(b_unst_cdf)
		tdiv0_pdf_curves.append(b_tdiv0_pdf); tdiv0_cdf_curves.append(b_tdiv0_cdf)
		tdd_pdf_curves.append(b_tdd_pdf); tdd_cdf_curves.append(b_tdd_cdf)
		tdie_pdf_curves.append(b_tdie_pdf); tdie_cdf_curves.append(b_tdie_cdf)

		# Calculate model prediction for each set of parameter
		b_model = Cyton2Model(hts, b_N0, mgen, DT, nreps)
		b_extrapolate = b_model.extrapolate(times, b_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
		b_ext_total_live_cells.append(b_extrapolate['ext']['total_live_cells'])
		b_ext_total_cohorts = np.sum(np.transpose(b_extrapolate['ext']['cells_gen']) * np.power(2.,-gens), axis=1)
		b_ext_cells_per_gen.append(b_extrapolate['ext']['cells_gen'])
		b_hts_total_live_cells.append(b_extrapolate['hts']['total_live_cells'])
		b_hts_cells_per_gen.append(b_extrapolate['hts']['cells_gen'])

		_b_cohorts = np.transpose(b_extrapolate['ext']['cells_gen']) * np.power(2.,-gens)  # compute MDN predictions
		_b_weighted = _b_cohorts * gens
		b_ext_mdn = np.sum(_b_weighted, axis=1) / b_ext_total_cohorts

		_b_cohorts_ex0 = np.transpose(b_extrapolate['ext']['cells_gen'][1:]) * np.power(2.,-gens[1:])  # compute MDN predictions excluding gen0
		_b_weighted_ext0 = _b_cohorts_ex0 * gens[1:]
		b_ext_mdn_ex0 = np.sum(_b_weighted_ext0, axis=1) / np.sum(np.transpose(b_extrapolate['ext']['cells_gen'][1:]) * np.power(2.,-gens[1:]), axis=1)

		conf['unst_pdf'].append(b_unst_pdf); conf['unst_cdf'].append(b_unst_cdf)
		conf['tdiv0_pdf'].append(b_tdiv0_pdf); conf['tdiv0_cdf'].append(b_tdiv0_cdf)
		conf['tdd_pdf'].append(b_tdd_pdf); conf['tdd_cdf'].append(b_tdd_cdf)
		conf['tdie_pdf'].append(b_tdie_pdf); conf['tdie_cdf'].append(b_tdie_cdf)
		conf['ext_total_cohorts'].append(b_ext_total_cohorts)
		conf['ext_total_live_cells'].append(b_ext_total_live_cells); conf['ext_cells_per_gen'].append(b_ext_cells_per_gen)
		conf['hts_total_live_cells'].append(b_hts_total_live_cells); conf['hts_cells_per_gen'].append(b_hts_cells_per_gen)
		conf['ext_mdn'].append(b_ext_mdn)
		conf['ext_mdn_ex0'].append(b_ext_mdn_ex0)

		tmp_N0.append(b_N0)

	# Calculate 95% confidence bands on PDF, CDF and model predictions
	for obj in conf:
		stack = np.vstack(conf[obj])
		conf[obj] = conf_iterval(stack, RGS)
	
	# 95% confidence interval on each parameter values
	err_mUns, err_sUns = conf_iterval(boots['mUns'], RGS), conf_iterval(boots['sUns'], RGS)
	err_mDiv0, err_sDiv0 = conf_iterval(boots['mDiv0'], RGS), conf_iterval(boots['sDiv0'], RGS)
	err_mDD, err_sDD = conf_iterval(boots['mDD'], RGS), conf_iterval(boots['sDD'], RGS)
	err_mDie, err_sDie = conf_iterval(boots['mDie'], RGS), conf_iterval(boots['sDie'], RGS)
	err_m, err_p = conf_iterval(boots['m'], RGS), conf_iterval(boots['p'], RGS)
	err_N0 = conf_iterval(tmp_N0, RGS)	# AGAIN, NOT A REAL PARAMETER

	save_best_fit = pd.DataFrame(
	data={"best-fit": [mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p, N0],
		"low95": [mUns-err_mUns[0], sUns-err_sUns[0], 
					mDiv0-err_mDiv0[0], sDiv0-err_sDiv0[0], 
					mDD-err_mDD[0], sDD-err_sDD[0], 
					mDie-err_mDie[0], sDie-err_sDie[0], 
					m-err_m[0], p-err_p[0], N0-err_N0[0]],
		"high95": [err_mUns[1]-mUns, err_sUns[1]-sUns, 
					err_mDiv0[1]-mDiv0, err_sDiv0[1]-sDiv0, 
					err_mDD[1]-mDD, err_sDD[1]-sDD, 
					err_mDie[1]-mDie, err_sDie[1]-sDie, 
					err_m[1]-m, err_p[1]-p, err_N0[1]-N0],
		"vary": np.append([params[p].vary for p in params], ["False"])}, 
	index=["mUns", "sUns", "mDiv0", "sDiv0", "mDD", "sDD", "mDie", "sDie", "m", "p", "N0"])

	excel_path = f"./out/CD4/{key}_{condition}_result.xlsx"
	with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
		save_best_fit.to_excel(writer, sheet_name="pars")
		boots.to_excel(writer, sheet_name="boot")

	# Get extrapolation: 1. for given time range t \in [t0, tf]; 2. at harvested time points
	model = Cyton2Model(hts, N0, mgen, DT, nreps)
	extrapolate = model.extrapolate(times, best_fit)  # get extrapolation for all "times" (discretised) and at harvested timepoints
	ext_total_live_cells = extrapolate['ext']['total_live_cells']
	ext_cells_per_gen = extrapolate['ext']['cells_gen']
	# hts_total_live_cells = extrapolate['hts']['total_live_cells']
	hts_cells_per_gen = extrapolate['hts']['cells_gen']

	# Calculate PDF and CDF
	tdiv0_pdf, tdiv0_cdf = lognorm_pdf(times, mDiv0, sDiv0), lognorm_cdf(times, mDiv0, sDiv0)
	tdd_pdf, tdd_cdf = lognorm_pdf(times, mDD, sDD), lognorm_cdf(times, mDD, sDD)
	tdie_pdf, tdie_cdf = lognorm_pdf(times, mDie, sDie), lognorm_cdf(times, mDie, sDie)
	### FIG 1: SUMMARY PLOT
	fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharex=True)
	fig1.suptitle(f"[{key}][{condition}] Summary")

	## MEAN DIVISION NUMBER
	ax1[0,0].set_ylabel("MDN")
	ax1[0,0].set_xlabel("Time (hour)")
	mdn_tps = []
	mdns = []
	mdns_ex0 = []
	for itpt, ht in enumerate(hts):
		for irep in range(all_reps[itpt]):
			mdn_tps.append(ht)
			_cgens = np.array(df['cgens']['rep'][icnd][itpt][irep])
			_cohorts = _cgens / np.power(2., gens)
			weighted = _cohorts * gens
			_mdn = np.sum(weighted) / np.sum(_cohorts)
			mdns.append(_mdn)

			_cohorts_ex0 = _cgens[1:] / np.power(2., gens[1:])
			weighted_ex0 = _cohorts_ex0 * gens[1:]
			_mdn_ex0 = np.sum(weighted_ex0) / np.sum(_cohorts_ex0)
			mdns_ex0.append(_mdn_ex0)
	ax1[0,0].plot(mdn_tps, mdns, 'r.', label='data')
	ax1[0,0].plot(mdn_tps, mdns_ex0, '.', color='navy', mfc='none', label='data (ex. gen0)')
	_ext_cohorts = np.transpose(ext_cells_per_gen) * np.power(2.,-gens)
	_ext_weighted = _ext_cohorts * gens
	ext_mdn = np.sum(_ext_weighted, axis=1) / np.sum(np.transpose(ext_cells_per_gen) * np.power(2.,-gens), axis=1)
	ax1[0,0].plot(times, ext_mdn, 'k-', label='model')
	ax1[0,0].fill_between(times, conf['ext_mdn'][0], conf['ext_mdn'][1], fc='k', ec=None, alpha=0.3)
	## MDN excluding gen.0
	_ext_cohorts_ex0 = np.transpose(ext_cells_per_gen[1:]) * np.power(2.,-gens[1:])
	_ext_weighted_ext0 = _ext_cohorts_ex0 * gens[1:]
	ext_mdn_ex0 = np.sum(_ext_weighted_ext0, axis=1) / np.sum(np.transpose(ext_cells_per_gen[1:]) * np.power(2.,-gens[1:]), axis=1)
	ax1[0,0].plot(times, ext_mdn_ex0, '--', color='navy', label='model (ex. gen0)')
	ax1[0,0].fill_between(times, conf['ext_mdn_ex0'][0], conf['ext_mdn_ex0'][1], fc='navy', ec=None, alpha=0.3)
	ax1[0,0].set_ylim(bottom=0)
	ax1[0,0].legend(fontsize=9, frameon=True)

	## TOTAL COHORT NUMBER
	ax1[0,1].set_ylabel("Cohort number")
	tps_incl, total_cohorts_incl = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(all_reps[itpt]):
			tps_incl.append(ht)
			total_cohorts_incl.append(np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep]))
	ext_total_cohorts = np.sum(np.transpose(ext_cells_per_gen) * np.power(2.,-gens), axis=1)
	ax1[0,1].plot(tps_incl, total_cohorts_incl, 'r.', label='data')
	ax1[0,1].plot(times, ext_total_cohorts, 'k-', label='model')
	ax1[0,1].fill_between(times, conf['ext_total_cohorts'][0], conf['ext_total_cohorts'][1], fc='k', ec=None, alpha=0.3)
	ax1[0,1].set_ylim(bottom=0)
	ax1[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1[0,1].yaxis.major.formatter._useMathText = True
	ax1[0,1].legend(fontsize=9, frameon=True)


	## PROBABILITY DISTRIBUTION FUNCTION
	ax1[1,0].set_title(f"$b={m:.2f}h \pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$")
	ax1[1,0].set_ylabel("Density")
	label_Tdiv0 = f"$T_{{div}}^0 \sim \mathcal{{LN}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.3f} \pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}})$"
	label_Tdd = f"$T_{{dd}} \sim \mathcal{{LN}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}})$"
	label_Tdie = f"$T_{{die}} \sim \mathcal{{LN}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}})$"
	ax1[1,0].plot(times, tdiv0_pdf, color='blue', ls='-', label=label_Tdiv0)
	ax1[1,0].fill_between(times, conf['tdiv0_pdf'][0], conf['tdiv0_pdf'][1], fc='blue', ec=None, alpha=0.5)
	ax1[1,0].plot(times, tdd_pdf, color='green', ls='-', label=label_Tdd)
	ax1[1,0].fill_between(times, conf['tdd_pdf'][0], conf['tdd_pdf'][1], fc='green', ec=None, alpha=0.5)
	ax1[1,0].plot(times, -tdie_pdf, color='red', ls='-', label=label_Tdie)
	ax1[1,0].fill_between(times, -conf['tdie_pdf'][0], -conf['tdie_pdf'][1], fc='red', ec=None, alpha=0.5)
	ax1[1,0].set_yticklabels(np.round(np.abs(ax1[0,0].get_yticks()), 5))  # remove negative y-tick labels
	ax1[1,0].legend(fontsize=9, frameon=True)

	## TOTAL CELL NUMBERS 
	ax1[1,1].set_title(f"$N_0 = {N0:.1f}\pm_{{{N0-err_N0[0]:.1f}}}^{{{err_N0[1]-N0:.1f}}}$")
	ax1[1,1].set_ylabel("Cell number")
	ax1[1,1].set_xlabel("Time (hour)")
	tps, total_cells = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(all_reps[itpt]):
			tps.append(ht)
			total_cells.append(df['cells']['rep'][icnd][itpt][irep])
	ax1[1,1].plot(tps, total_cells, 'r.')
	ax1[1,1].plot(times, ext_total_live_cells, 'k-', lw=1)
	ax1[1,1].fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc='k', ec=None, alpha=0.3)
	cp = sns.hls_palette(mgen+1, l=0.4, s=0.5)
	for igen in range(mgen+1):
		# ax1[1,1].errorbar(hts, np.transpose(df['cgens']['avg'][icnd])[igen], yerr=np.transpose(df['cgens']['sem'][icnd])[igen], c=cp[igen], fmt='.', ms=5, label=f"Gen {igen}")
		ax1[1,1].errorbar(hts, np.transpose(df['cgens']['avg'][icnd])[igen], yerr=np.transpose(df['cgens']['sem'][icnd])[igen], c=cp[igen], fmt='.', ms=5, label=f"Gen {igen}")
		ax1[1,1].plot(times, ext_cells_per_gen[igen], c=cp[igen])
		ax1[1,1].fill_between(times, conf['ext_cells_per_gen'][0][igen], conf['ext_cells_per_gen'][1][igen], fc=cp[igen], ec=None, alpha=0.5)
	ax1[1,1].set_ylim(bottom=0)
	ax1[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1[1,1].yaxis.major.formatter._useMathText = True
	# ax1[1,1].symlog()
	ax1[1,1].legend(fontsize=9, frameon=True)
	for ax in ax1.flat:
		ax.set_xlim(t0, max(times))
	# fig1.subplots_adjust(hspace=0, wspace=0)
	fig1.tight_layout(rect=(0.0, 0.0, 1, 1))

	### FIG 2: CELL NUMBERS PER GENERATION AT HARVESTED TIME POINTS
	if len(hts) <= 6: nrows, ncols = 2, 3
	elif 6 < len(hts) <= 9: nrows, ncols = 3, 3
	else: nrows, ncols = 4, 3

	fig2 = plt.figure()
	# fig2.suptitle(f"[{condition}] Cell numbers per generation at harvested time")
	fig2.text(0.5, 0.04, "Generations", ha='center', va='center')
	fig2.text(0.02, 0.5, "Cell number", ha='center', va='center', rotation=90)
	axes = []  # store axis
	for itpt, ht in enumerate(hts):
		ax2 = plt.subplot(nrows, ncols, itpt+1)
		ax2.set_axisbelow(True)
		ax2.plot(gens, hts_cells_per_gen[itpt], 'o-', c='k', ms=5, label='model')
		ax2.fill_between(gens, conf['hts_cells_per_gen'][0][itpt], conf['hts_cells_per_gen'][1][itpt], fc='k', ec=None, alpha=0.3)
		for irep in range(all_reps[itpt]):
			ax2.plot(gens, df['cgens']['rep'][icnd][itpt][irep], 'r.', label='data')
		ax2.set_xticks(gens)
		ax2.annotate(f"{ht}h", xy=(0.75, 0.85), xycoords='axes fraction')
		ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		ax2.yaxis.major.formatter._useMathText = True
		if itpt not in [len(hts)-3, len(hts)-2, len(hts)-1]:
			# ax2.get_xaxis().set_ticks([])
			ax2.set_xticklabels([])
		if itpt not in [0, 3, 6, 9, 12]:
			# ax2.get_yaxis().set_ticks([])
			ax2.set_yticklabels([])
		ax2.spines['right'].set_visible(True)
		ax2.spines['top'].set_visible(True)
		ax2.grid(True, ls='--')
		axes.append(ax2)
	max_ylim = 0
	for ax in axes:
		_, ymax = ax.get_ylim()
		if max_ylim < ymax:
			max_ylim = ymax
	for ax in axes:
		ax.set_ylim(top=max_ylim)
	# handles, labels = ax.get_legend_handles_labels()
	# fig2.legend(handles=[handles[1], handles[0]], labels=[labels[1], labels[0]], ncol=2, bbox_to_anchor=(1,1))
	handles, labels = [], []
	for ax in fig2.axes:
		hs, ls = ax.get_legend_handles_labels()
		for h, l in zip(hs, ls):
			if l not in labels:
				handles.append(h)
				labels.append(l)
	fig2.legend(handles=handles, labels=labels, ncol=3, bbox_to_anchor=(1,1))
	fig2.tight_layout(rect=(0.02, 0.03, 1, 1))
	fig2.subplots_adjust(wspace=0.05, hspace=0.15)


	### FIG 3: DISTRIBUTION OF BOOTSTRAP SAMPLES 
	alpha = (1. - RGS/100.)/2
	quantiles = boots[['mUns', 'sUns', 'mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'm', 'p', 'N0']].quantile([alpha, alpha + RGS/100.], numeric_only=True, interpolation='nearest')
	
	titles = ["Time to first division ($T_{div}^0$)", "Time to first division ($T_{div}^0$)", "Time to division destiny ($T_{dd}$)", "Time to division destiny ($T_{dd}$)", "Time to death ($T_{die}$)", "Time to death ($T_{die}$)", "Subsequent division time", "Starting cell numbers"]
	xlabs = ["median ($m_{div}^0$)", "shape ($s_{div}^0$)", "median ($m_{dd}$)", "shape ($s_{dd}$)", "median ($m_{die}$)", "shape ($s_{die}$)", "$t_{div}$ (hour)", "$N_0$"]
	colors = ['blue', 'blue', 'green', 'green', 'red', 'red', 'navy', 'k']
	fig3, ax3 = plt.subplots(nrows=4, ncols=2, figsize=(9, 8))
	fig3.suptitle(f"[{condition}] Bootstrap marginal distribution")
	ax3 = ax3.flat
	for i, obj in enumerate(boots.drop(['mUns', 'sUns', 'p'], axis=1)):
		if obj in list(best_fit.keys()):
			best = best_fit[obj]
		else:
			best = boots[obj].mean()
		b_sample = boots[obj].to_numpy()
		l_quant, h_quant = quantiles.iloc[0][obj], quantiles.iloc[1][obj]

		ax3[i].set_title(titles[i])
		ax3[i].axvline(best, ls='-', c='k', label=f"best-fit={best:.2f}")
		ax3[i].axvline(l_quant, ls=':', c='red', label=f"lo={l_quant:.2f}")
		ax3[i].axvline(h_quant, ls='-.', c='red', label=f"hi={h_quant:.2f}")
		sns.distplot(b_sample, kde=False, hist_kws=dict(ec='k', lw=1), color=colors[i], ax=ax3[i])
		ax3[i].set_xlabel(xlabs[i])
		ax3[i].legend(fontsize=9, loc='upper right')
	fig3.tight_layout(rect=(0.01, 0, 1, 1))
	fig3.subplots_adjust(wspace=0.1, hspace=0.83)

	with PdfPages(f"./out/CD4/{key}_{condition}.pdf") as pdf:
		pdf.savefig(fig1)
		pdf.savefig(fig2)
		pdf.savefig(fig3)
	
if __name__ == "__main__":
	start = time.time()
	print('> No. of BOOTSTRAP ITERATIONS: {0}'.format(ITER_BOOTS))
	print('> No. of SEARCH ITERATIONS for CYTON FITTING: {0}'.format(ITER_SEARCH))

	########### DATA ###########
	DATA_FILES = [
		'2023_07_03-CD4 TCM (rev).xlsx',
		'2023_07_03-CD4 Block (rev).xlsx',
		'2023_07_03-CD4 Block+IL-2 (rev).xlsx'
	]
	KEYS = [os.path.splitext(os.path.basename(data_key))[0] for data_key in DATA_FILES]
	df = parse_data('./data', DATA_FILES)

	inputs = []
	for key in KEYS:
		reader = df[key]['reader']
		for icnd, cond in enumerate(reader.condition_names):
			inputs.append((key, df[key], reader, icnd))
				
	tqdm.tqdm.set_lock(mp.RLock())  # for managing output contention
	p = mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
	with tqdm.tqdm(total=len(inputs), desc="Data Files", position=0) as pbar:
		for i, _ in enumerate(p.imap_unordered(fit, inputs)):
			pbar.update()
	p.close()
	p.join()

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f"> DONE FITTING ! {now}")
	print("> Elapsed Time = {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))