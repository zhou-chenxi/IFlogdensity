from IFlogdensity.contam_sm_de import ContamSMDensityEstimate

from dekef.base_density import *
from dekef.kernel_function import *
from dekef.scorematching_common_functions import *
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt


def plot_IF_1d_params(x_limit, y_limit=None, plot_pts_cnt=2000, figsize=(10, 10),
					  IF_color='tab:blue', linewidth=2.0,
					  rugplot_data_color='tab:blue', rugplot_contam_data_color='red',
					  title_fontsize=20, label_fontsize=15, tick_fontsize=10, info_fontsize=16,
					  contam_data_marker_color='tab:purple', contam_data_marker_alpha=0.5):
	
	"""
	Specifies and returns the plotting parameters used in the functions plot_IF_logdensity_1d, plot_IF_natparam_1d,
	or plot_IF_natparam_limit_1d in the class SMInfluenceFunction.
	
	Parameters
	----------
	x_limit : tuple
		The tuple to specify the plotting domain of the density estimate.
		Must be of length 2. Both components must be finite numbers.
	
	y_limit : tuple or None, optional
		The tuple to specify the domain of the plot of density estimate in the vertical axis; default is None.
		Must be of length 2. Both components must be finite numbers if not None.
	
	plot_pts_cnt : int, optional
		The number of points to be evaluated along the plot_domain to make a plot; default is 2000.
	
	figsize : typle, optional
		The size of the plot; default is (10, 10).
	
	IF_color : str, optional
		The color to plot the influence function; default is 'tab:blue'.
	
	linewidth : float, optional
		The line width parameter to plot the influence function; default is 2.0.
	
	rugplot_data_color : str, optional
		The color of the rug plot to indicate the location of the uncontaminated observations;
		default is 'tab:blue'.
		
	rugplot_contam_data_color : str, optional
		The color of the rug plot to indicate the location of the contaminated observation;
		default is 'red'.
	
	title_fontsize : int, optional
		The font size of the title of the plot; default is 20.
	
	label_fontsize : int, optional
		The font size of the label of the plot; default is 15.
		
	tick_fontsize : int, optional
		The font size of the tick label of the plot; default is 10.
	
	info_fontsize : int, optional
		The font size of the information of the plot to show where the contaminated observation is located;
		default is 16.
	
	contam_data_marker_color : str, optional
		The color of the vertical line to indicate the location of the contaminated observation;
		default is 'tab:purple'.
	
	contam_data_marker_alpha : float, optional
		The alpha value of the vertical line to indicate the location of the contaminated observation;
		default is 0.5.
	
	Returns
	-------
	dict
		A dict containing all the plotting parameter inputs.
		
	"""
	
	output = {'x_limit': x_limit,
			  'y_limit': y_limit,
			  'plot_pts_cnt': plot_pts_cnt,
			  'figsize': figsize,
			  'IF_color': IF_color,
			  'linewidth': linewidth,
			  'rugplot_data_color': rugplot_data_color,
			  'rugplot_contam_data_color': rugplot_contam_data_color,
			  'title_fontsize': title_fontsize,
			  'label_fontsize': label_fontsize,
			  'tick_fontsize': tick_fontsize,
			  'info_fontsize': info_fontsize,
			  'contam_data_marker_color': contam_data_marker_color,
			  'contam_data_marker_alpha': contam_data_marker_alpha}
	
	return output


class SMInfluenceFunction:
	
	"""
	A class to compute and visualize the influence function of the logarithm of score matching density estimate.
	
	...
	
	Attributes
	----------
	contam_density : ContamSMDensityEstimate object
		An object returned from the ContamSMDensityEstimate class, where the contam_weight therein must be strictly
		positive.
	
	uncontam_density : ContamSMDensityEstimate object
		An object returned from the ContamSMDensityEstimate class, where the contam_weight therein must be 0.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
		__type__ must be 'base_density'.
	
	Methods
	-------
	eval_IF_logdensity(new_data)
		Evaluates the influence function of the logarithm of the score matching density estimate at new_data.
	
	eval_IF_natparam(new_data)
		Evaluates the influence function of the natural parameter
		in the score matching density estimate at new_data.
	
	eval_IF_natparam_limit(new_data)
		Evaluates the influence function of the limiting natural parameter
		in the score matching density estimate at new_data
		as the contaminated observation goes to infinity.
	
	eval_IF_natparam_norm()
		Evaluates the RKHS norm of the influence function of the natural parameter
		in the score matching density estimate.
	
	eval_IF_natparam_limit_norm_1d()
		Evaluates the RKHS norm of the influence function of the limiting natural parameter
		in the score matching density estimate.
	
	plot_IF_logdensity_1d(plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None)
		Computes and plots the influence function of the logarithm of
		the score matching density estimate over a specified bounded interval.
	
	plot_IF_natparam_1d(plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None)
		Computes and plots the influence function of the natural parameter
		in the score matching density estimate over a specified bounded interval.
	
	plot_IF_natparam_limit_1d(plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None)
		Computes and plots the influence function of the limiting natural parameter
		in the score matching density estimate over a specified bounded interval.
	
	"""
	
	def __init__(self, data, contam_data, contam_weight, penalty_param, base_density,
				 r1=1.0, r2=0., c=0., bw=1.0, kernel_type='gaussian_poly2'):
		
		"""
		Parameters
		----------
		
		data : numpy.ndarray
			The array of observations whose probability density function is to be estimated.

		contam_data : numpy.ndarray
			The array of contaminated observation.

		contam_weight : float
			The weight of contamination.

		penalty_param : float
			The penalty parameter. Must be strictly positive.

		base_density : base_density object
			The base density function used to estimate the probability density function.

		r1 : float, optional
			The multiplicative constant associated with the Gaussian kernel function or the rational quadratic kernel
			function, depending on kernel_type; default is 1.

		r2 : float, optional
			The multiplicative constant associated with the polynomial kernel function of degree 2; default is 0.

		c : float, optional
			The non-homogenous additive constant in the polynomial kernel function of degree 2; default is 0.

		bw : float, optional
			The bandwidth parameter in the Gaussian kernel function or the rational quadratic kernel function,
			depending on kernel_type; default is 1.

		kernel_type : str, optional
			The type of the kernel function used to estimate the probability density function;
			must be one of 'gaussian_poly2' and 'rationalquad_poly2'; default is 'gaussian_poly2'.
			
		"""
		
		# construct the contaminated density estimate
		self.contam_density = ContamSMDensityEstimate(
			data=data,
			contam_data=contam_data,
			contam_weight=contam_weight,
			penalty_param=penalty_param,
			base_density=base_density,
			r1=r1,
			r2=r2,
			c=c,
			bw=bw,
			kernel_type=kernel_type)
		
		# construct the uncontaminated density estimate
		self.uncontam_density = ContamSMDensityEstimate(
			data=data,
			contam_data=contam_data,
			contam_weight=0.,
			penalty_param=penalty_param,
			base_density=base_density,
			r1=r1,
			r2=r2,
			c=c,
			bw=bw,
			kernel_type=kernel_type)
		
		self.base_density = base_density
	
	def eval_IF_logdensity(self, new_data):
		
		"""
		Evaluates the influence function of the logarithm of the score matching density estimate at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			An array of data points at which the influence function of
			the logarithm of the score matching density estimate is to be evaluated.
	
		Returns
		-------
		numpy.ndarray
			The value of the influence function of the logarithm of
			the score matching density estimate at new_data.
			
		"""
		
		if self.contam_density.contam_weight == 0.:
			raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
		
		# contaminated log-density function part
		contam_results = self.contam_density.log_density(new_data=new_data)
		contam_logden = contam_results
		
		# uncontaminated log-density function part
		uncontam_results = self.uncontam_density.log_density(new_data=new_data)
		uncontam_logden = uncontam_results
		
		# apply the finite difference method to approximate the influence function
		output = (contam_logden - uncontam_logden) / self.contam_density.contam_weight
		
		return output
	
	def eval_IF_natparam(self, new_data):
		
		"""
		Evaluates the influence function of the natural parameter
		in the score matching density estimate at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			An array of data points at which the influence function of
			the natural parameter in the score matching density estimate is to be evaluated.
	
		Returns
		-------
			The value of the influence function of the natural parameter in
			the score matching density estimate at new_data.
			
		"""
		
		if self.contam_density.contam_weight == 0.:
			raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
		
		# contaminated natural parameter part
		contam_coef = self.contam_density.coef()
		contam_results = self.contam_density.natural_param(new_data=new_data, coef=contam_coef)
		contam_natparam = contam_results
		
		# uncontaminated log-density function part
		uncontam_coef = self.uncontam_density.coef()
		uncontam_results = self.uncontam_density.natural_param(new_data=new_data, coef=uncontam_coef)
		uncontam_natparam = uncontam_results
		
		# apply the finite difference method to approximate the influence function
		output = (contam_natparam - uncontam_natparam) / self.contam_density.contam_weight
		
		return output
	
	def eval_IF_natparam_limit(self, new_data):
		
		"""
		Evaluates the influence function of the limiting natural parameter
		in the score matching density estimate at new_data
		as the contaminated observation goes to infinity.
		This function only works when the underlying data is 1-dimensional.
		
		Parameters
		----------
		new_data : numpy.ndarray
			An array of data points at which the influence function of
			the limiting natural parameter in the score matching density estimate is to be evaluated.
	
		Returns
		-------
			The value of the influence function of the limiting natural parameter in
			the score matching density estimate at new_data.
			
		"""
		
		assert self.contam_density.d == 1, 'The function eval_IF_natparam_limit only works for 1 dimensional data.'
	
		pen_param = self.contam_density.penalty_param
		K11 = self.uncontam_density.matrix_K11()
		K13 = self.uncontam_density.matrix_K13()
		N = self.uncontam_density.N
		d = self.uncontam_density.d
		
		K11_inv = np.linalg.inv(K11 + N * pen_param * np.eye(N * d))
		prod1 = np.matmul(K11, K11_inv)
		prod2 = np.matmul(prod1, K13) - 2 * K13
		prod3 = np.matmul(K11_inv, prod2)
		gamma_coef = np.vstack((- prod3 / pen_param, - 1. / pen_param))
		
		# partial_u k (X_j, \cdot) part
		f_matrix = kernel_partial10_hatz(
			data=self.uncontam_density.data,
			new_data=new_data,
			kernel_function=self.uncontam_density.kernel_function_data,
			base_density=self.base_density)
		part1 = np.matmul(f_matrix.T, gamma_coef).flatten()
		
		# z_{delta_y} part
		part2 = (kernel_partial10_hatz(
			data=self.contam_density.contam_data,
			new_data=new_data,
			kernel_function=self.contam_density.kernel_function_contam_data,
			base_density=self.base_density)[[-1]]).flatten()
	
		output = part1 + part2 / pen_param
		return output
	
	def eval_IF_natparam_norm(self):
		
		"""
		Evaluates the RKHS norm of the influence function of the natural parameter
		in the score matching density estimate.
		
		Returns
		-------
			The value of the RKHS norm of the influence function of the natural parameter in
			the score matching density estimate.
			
		"""
		
		N, d = self.contam_density.N, self.contam_density.d
		large_K = self.contam_density.matrix_K()
		
		# compute alpha coefficient vector
		pen_param = self.contam_density.penalty_param
		
		K11 = self.contam_density.matrix_K11()
		K12 = self.contam_density.matrix_K12()
		K13 = self.contam_density.matrix_K13()
		K14 = self.contam_density.matrix_K14()
		
		K21 = self.contam_density.matrix_K21()
		K23 = self.contam_density.matrix_K23()
		
		K11_inv = np.linalg.inv(K11 + N * pen_param * np.eye(N * d))
		
		# coef with data
		coef1_1 = - 1. / pen_param * np.matmul(K11_inv, np.matmul(np.matmul(K11, K11_inv), K13) - 2. * K13 + K14)
		coef1_2 = - 1. / pen_param ** 2 * np.matmul(np.matmul(np.matmul(np.matmul(K11_inv, K12), K21), K11_inv), K13)
		coef1_3 = 1. / pen_param ** 2 * np.matmul(np.matmul(K11_inv, K12), K23)
		
		coef1 = coef1_1 + coef1_2 + coef1_3
		
		# coef with contam_data
		coef2 = 1. / pen_param ** 2 * np.matmul(np.matmul(K21, K11_inv), K13) - 1. / pen_param ** 2 * K23
		
		coef = np.vstack((coef1,
						  coef2,
						  - 1. / pen_param,
						  1. / pen_param)).reshape(-1, 1)
		
		output = np.matmul(coef.T, (np.matmul(large_K, coef))).item()
		
		return np.sqrt(output)
	
	def eval_IF_natparam_limit_norm_1d(self):
		
		"""
		Evaluates the RKHS norm of the influence function of the limiting natural parameter
		in the score matching density estimate.
		This function only works when the underlying data is 1-dimensional.
		
		Returns
		-------
			The value of the RKHS norm of the influence function of the limiting natural parameter in
			the score matching density estimate.

		"""
		
		N, d = self.contam_density.N, self.contam_density.d
		assert d == 1, f'The function eval_IF_natparam_limit_norm_1d only works for 1-dimensional data, ' \
					   f'but got {d}-dimensional data.'
		
		pen_param = self.contam_density.penalty_param
		K11 = self.contam_density.matrix_K11()
		K13 = self.contam_density.matrix_K13()
		K11_inv = np.linalg.inv(K11 + N * pen_param * np.eye(N * d))
		prod1 = np.matmul(K11, K11_inv)
		prod2 = np.matmul(prod1, K13) - 2. * K13
		prod3 = np.matmul(K11_inv, prod2)
		gamma_coef = - prod3 / pen_param
		
		part1 = np.matmul(gamma_coef.T, (np.matmul(K11, gamma_coef))).item()
		
		# norm{z_{F_n}}^2 / pen_param ** 2
		part2 = self.contam_density.matrix_K33().item() / pen_param ** 2
		
		# norm{z_{delta_y}}^2 / pen_param ** 2
		if self.base_density.name == 'Gamma':
			mu_limit = - 1. / self.base_density.scale
		elif self.base_density.name == 'Lognormal':
			mu_limit = 0.
		elif self.base_density.name == 'Exponential':
			mu_limit = -1. / self.base_density.scale
		elif self.base_density.name == 'Uniform':
			mu_limit = 0.
		else:
			raise NotImplementedError(f'The base density used is {self.base_density.name}, '
									  f'which has not been implemented.')
		
		if self.contam_density.kernel_type == 'gaussian_poly2':
			
			if self.contam_density.kernel_function_data.r2 != 0.:
				raise ValueError('In order to use the function eval_IF_natparam_limit_norm_1d, must set r2 to be 0.')
			
			ker11 = self.contam_density.kernel_function_data.r1 * 1. / self.contam_density.kernel_function_data.bw ** 2
			ker12 = 0.
			ker22 = self.contam_density.kernel_function_data.r1 * 3. / self.contam_density.kernel_function_data.bw ** 4
		
		elif self.contam_density.kernel_type == 'rationalquad_poly2':
			
			if self.contam_density.kernel_function_data.r2 != 0.:
				raise ValueError('In order to use the function eval_IF_natparam_limit_norm_1d, must set r2 to be 0.')
			
			ker11 = self.contam_density.kernel_function_data.r1 * 2. / self.contam_density.kernel_function_data.bw ** 2
			ker12 = 0.
			ker22 = self.contam_density.kernel_function_data.r1 * 24. / self.contam_density.kernel_function_data.bw ** 4
		
		part3 = (ker22 + 2. * mu_limit * ker12 + mu_limit ** 2 * ker11) / pen_param ** 2
		
		# inner product between partial_u k (X_i, .) and z_{F_n}
		part4 = 2. * np.sum(K13.flatten() * gamma_coef.flatten()) / pen_param
		
		output = part1 + part2 + part3 - part4
		
		return np.sqrt(output)
	
	def plot_IF_logdensity_1d(self, plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None):
		
		"""
		Computes and plots the influence function of the logarithm of
		the score matching density estimate over a specified bounded interval.
		This functional only works when the underlying data is 1-dimensional.
		
		Parameters
		----------
		plot_kwargs : dict
			The dict containing plotting parameters returned from the function plot_IF_1d_params.
		
		x_label : str
			The label of the horizontal axis.
		
		save_plot : bool, optional
			Whether to save the plot of the influence function to a local file; default is False.

		save_dir : str, optional
			The directory path to which the plot of the influence function is saved;
			only works when save_plot is set to be True. Default is None.
	
		save_filename : str, optional
			The file name for the plot of the influence function saved as a local file;
			only works when save_plot is set to be True. Default is None.
		
		Returns
		-------
		dict
			A dictionary of 'x_vals', the values of the horizontal axis for plotting, and
			'IF_vals', the values of the vertical axis for plotting.
		
		"""
		
		# check the dimensionality
		if self.contam_density.d != 1:
			raise ValueError(
				f'In order to plot the influence function, the dimensionality of data must be 1, but got {self.contam_density.d}.')
		
		if len(plot_kwargs['x_limit']) != 2:
			raise ValueError("The length of x_limit in plot_kwargs must be 2.")
		
		if np.inf in plot_kwargs['x_limit'] or -np.inf in plot_kwargs['x_limit']:
			raise ValueError("The 'x_limit' in plot_kwargs contains non-finite values.")
		
		plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
		plot_pts_cnt = plot_kwargs['plot_pts_cnt']
		
		# prepare the data for plotting
		new_data = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)
		result = self.eval_IF_logdensity(new_data)
		
		# make the plot
		fig = plt.figure(figsize=plot_kwargs['figsize'])
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height])
		
		plt.plot(new_data, result, color=plot_kwargs['IF_color'], linewidth=plot_kwargs['linewidth'])
		ax.set_title('Plot of the Influence Function', fontsize=plot_kwargs['title_fontsize'])
		ax.set_xlabel(x_label, fontsize=plot_kwargs['label_fontsize'])
		ax.set_ylabel('IF', fontsize=plot_kwargs['label_fontsize'])
		ax.set_xlim(plot_kwargs['x_limit'])
		if plot_kwargs['y_limit'] is None:
			ax.set_ylim(plot_kwargs['y_limit'])
		ax.tick_params(axis='both', labelsize=plot_kwargs['tick_fontsize'])
		
		# add plot information
		info = f'Add {self.contam_density.contam_data[0][0]}'
		ax.text(0.988, 0.988,
				info,
				fontsize=plot_kwargs['info_fontsize'],
				multialignment='left',
				horizontalalignment='right',
				verticalalignment='top',
				transform=ax.transAxes,
				bbox={'facecolor': 'none',
					  'boxstyle': 'Round, pad=0.2'})
		
		# mark the contaminated observation
		ax.axvline(self.contam_density.contam_data[0][0], 0, 1,
				   ls='--',
				   color=plot_kwargs['contam_data_marker_color'],
				   alpha=plot_kwargs['contam_data_marker_alpha'])
		
		# add rug plot
		rug_df = pd.DataFrame({'data': self.contam_density.data.flatten()})
		seaborn.rugplot(rug_df['data'], ax=ax, color=plot_kwargs['rugplot_data_color'])
		seaborn.rugplot(self.contam_density.contam_data[0], ax=ax, color=plot_kwargs['rugplot_contam_data_color'])
		
		if save_plot:
			plt.savefig(save_dir + save_filename + '.pdf')
		plt.show()
		
		return {'x_vals': new_data.flatten(), "IF_vals": result.flatten()}
	
	def plot_IF_natparam_1d(self, plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None):
		
		"""
		Computes and plots the influence function of the natural parameter
		in the score matching density estimate over a specified bounded interval.
		This functional only works when the underlying data is 1-dimensional.
		
		Parameters
		----------
		plot_kwargs : dict
			The dict containing plotting parameters returned from the function plot_IF_1d_params.
		
		x_label : str
			The label of the horizontal axis.
		
		save_plot : bool, optional
			Whether to save the plot of the influence function to a local file; default is False.

		save_dir : str, optional
			The directory path to which the plot of the influence function is saved;
			only works when save_plot is set to be True. Default is None.
	
		save_filename : str, optional
			The file name for the plot of the influence function saved as a local file;
			only works when save_plot is set to be True. Default is None.
		
		Returns
		-------
		dict
			A dictionary of 'x_vals', the values of the horizontal axis for plotting, and
			'IF_vals', the values of the vertical axis for plotting.
		 
		"""
		# check the dimensionality
		if self.contam_density.d != 1:
			raise ValueError(
				f'In order to plot the influence function, the dimensionality of data must be 1, but got {self.contam_density.d}.')
		
		if len(plot_kwargs['x_limit']) != 2:
			raise ValueError("The length of x_limit in plot_kwargs must be 2.")
		
		if np.inf in plot_kwargs['x_limit'] or -np.inf in plot_kwargs['x_limit']:
			raise ValueError("The 'x_limit' in plot_kwargs contains non-finite values.")
		
		plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
		plot_pts_cnt = plot_kwargs['plot_pts_cnt']
		
		# prepare the data for plotting
		new_data = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)
		result = self.eval_IF_natparam(new_data)
		
		# make the plot
		fig = plt.figure(figsize=plot_kwargs['figsize'])
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height])
		
		plt.plot(new_data, result, color=plot_kwargs['IF_color'], linewidth=plot_kwargs['linewidth'])
		ax.set_title('Plot of the Influence Function', fontsize=plot_kwargs['title_fontsize'])
		ax.set_xlabel(x_label, fontsize=plot_kwargs['label_fontsize'])
		ax.set_ylabel('IF', fontsize=plot_kwargs['label_fontsize'])
		ax.set_xlim(plot_kwargs['x_limit'])
		if plot_kwargs['y_limit'] is None:
			ax.set_ylim(plot_kwargs['y_limit'])
		ax.tick_params(axis='both', labelsize=plot_kwargs['tick_fontsize'])
		
		# add plot information
		info = f'Add {self.contam_density.contam_data[0][0]}'
		ax.text(0.988, 0.988,
				info,
				fontsize=plot_kwargs['info_fontsize'],
				multialignment='left',
				horizontalalignment='right',
				verticalalignment='top',
				transform=ax.transAxes,
				bbox={'facecolor': 'none',
					  'boxstyle': 'Round, pad=0.2'})
		
		# mark the contaminated observation
		ax.axvline(self.contam_density.contam_data[0][0], 0, 1,
				   ls='--',
				   color=plot_kwargs['contam_data_marker_color'],
				   alpha=plot_kwargs['contam_data_marker_alpha'])
		
		# add rug plot
		rug_df = pd.DataFrame({'data': self.contam_density.data.flatten()})
		seaborn.rugplot(rug_df['data'], ax=ax, color=plot_kwargs['rugplot_data_color'])
		seaborn.rugplot(self.contam_density.contam_data[0], ax=ax, color=plot_kwargs['rugplot_contam_data_color'])
		
		if save_plot:
			plt.savefig(save_dir + save_filename + '.pdf')
		plt.show()
		
		return {'x_vals': new_data.flatten(), "IF_vals": result.flatten()}
	
	def plot_IF_natparam_limit_1d(self, plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None):
		
		"""
		Computes and plots the influence function of the limiting natural parameter
		in the score matching density estimate over a specified bounded interval.
		This function only works when the underlying data is 1-dimensional.
		
		Parameters
		----------
		plot_kwargs : dict
			The dict containing plotting parameters returned from the function plot_IF_1d_params.
		
		x_label : str
			The label of the horizontal axis.
		
		save_plot : bool, optional
			Whether to save the plot of the influence function to a local file; default is False.

		save_dir : str, optional
			The directory path to which the plot of the influence function is saved;
			only works when save_plot is set to be True. Default is None.
	
		save_filename : str, optional
			The file name for the plot of the influence function saved as a local file;
			only works when save_plot is set to be True. Default is None.
		
		Returns
		-------
		dict
			A dictionary of 'x_vals', the values of the horizontal axis for plotting, and
			'IF_vals', the values of the vertical axis for plotting.
			
		"""
		
		# check the dimensionality
		if self.contam_density.d != 1:
			raise ValueError(
				f'In order to plot the influence function, the dimensionality of data must be 1, but got {self.contam_density.d}.')
		
		if len(plot_kwargs['x_limit']) != 2:
			raise ValueError("The length of x_limit in plot_kwargs must be 2.")
		
		if np.inf in plot_kwargs['x_limit'] or -np.inf in plot_kwargs['x_limit']:
			raise ValueError("The 'x_limit' in plot_kwargs contains non-finite values.")
		
		plot_domain = [[plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1]]]
		plot_pts_cnt = plot_kwargs['plot_pts_cnt']
		
		# prepare the data for plotting
		new_data = np.linspace(plot_domain[0][0], plot_domain[0][1], num=plot_pts_cnt).reshape(-1, 1)
		result = self.eval_IF_natparam_limit(new_data)
		
		# make the plot
		fig = plt.figure(figsize=plot_kwargs['figsize'])
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height])
		
		plt.plot(new_data, result, color=plot_kwargs['IF_color'], linewidth=plot_kwargs['linewidth'])
		ax.set_title('Plot of the Influence Function', fontsize=plot_kwargs['title_fontsize'])
		ax.set_xlabel(x_label, fontsize=plot_kwargs['label_fontsize'])
		ax.set_ylabel('IF', fontsize=plot_kwargs['label_fontsize'])
		ax.set_xlim(plot_kwargs['x_limit'])
		if plot_kwargs['y_limit'] is None:
			ax.set_ylim(plot_kwargs['y_limit'])
		ax.tick_params(axis='both', labelsize=plot_kwargs['tick_fontsize'])
		
		# add plot information
		info = f'Add {self.contam_density.contam_data[0][0]}'
		ax.text(0.988, 0.988,
				info,
				fontsize=plot_kwargs['info_fontsize'],
				multialignment='left',
				horizontalalignment='right',
				verticalalignment='top',
				transform=ax.transAxes,
				bbox={'facecolor': 'none',
					  'boxstyle': 'Round, pad=0.2'})
		
		# mark the contaminated observation
		ax.axvline(self.contam_density.contam_data[0][0], 0, 1,
				   ls='--',
				   color=plot_kwargs['contam_data_marker_color'],
				   alpha=plot_kwargs['contam_data_marker_alpha'])
		
		# add rug plot
		rug_df = pd.DataFrame({'data': self.contam_density.data.flatten()})
		seaborn.rugplot(rug_df['data'], ax=ax, color=plot_kwargs['rugplot_data_color'])
		seaborn.rugplot(self.contam_density.contam_data[0], ax=ax, color=plot_kwargs['rugplot_contam_data_color'])
		
		if save_plot:
			plt.savefig(save_dir + save_filename + '.pdf')
		plt.show()
		
		return {'x_vals': new_data.flatten(), "IF_vals": result.flatten()}
	