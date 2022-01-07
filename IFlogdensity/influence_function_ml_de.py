from IFlogdensity.contam_ml_de import *

from dekef.base_density import *
from dekef.kernel_function import *
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
	Specifies and returns the plotting parameters used in the functions plot_IF_logdensity_1d
	in the class MLInfluenceFunction.
	
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


class MLInfluenceFunction:
	
	"""
	A class to compute and plot the influence function of the logarithm of maximum likelihood density estimate.

	...

	Attributes
	----------
	contam_density : ContamMLDensityEstimate object
		An object returned from the ContamMLDensityEstimate class, where the contam_weight therein must be strictly
		positive.
	
	uncontam_density : ContamMLDensityEstimate object
		An object returned from the ContamMLDensityEstimate class, where the contam_weight therein must be 0.
	
	base_density : base_density object
		The base density function used to estimate the probability density function.
		__type__ must be 'base_density'.

	Methods
	-------
	eval_IF_logdensity(new_data, basis_type, optalgo_params, batchmc_params, grid_points=None,
					   random_seed=0, step_size_factor=1.0, algo='gd', print_error=True)
		Evaluates the influence functions of the logarithm of the maximum likelihood density estimate at new_data.

	plot_IF_logdensity_1d(basis_type, optalgo_params, batchmc_params, plot_kwargs, x_label,
						  grid_points=None, step_size_factor=None, random_seed=0, algo='gd',
						  print_error=True, save_plot=False, save_dir=None, save_filename=None)
		Computes and plots the influence function of the logarithm of
		the maximum likelihood density estimate over a specified bounded interval.

	"""
	
	def __init__(self, data, contam_data, contam_weight, penalty_param, base_density,
				 r1=1.0, r2=0., c=0., bw=1.0, kernel_type='gaussian_poly2'):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array of uncontaminated observations.
		
		contam_data : numpy.ndarray
			The contaminated observation.
		
		contam_weight : float
			The weight of self.contam_data; must always be between 0 and 1, inclusively.
		
		penalty_param : float
			The penalty parameter used to compute the penalized score matching density estimate;
			must be non-negative.
			
		base_density : base_density
			The base density function used to estimate the probability density function.
			__type__ must be 'base_density'.
		
		r1 : float
			The multiplicative coefficient associated with the Gaussian kernel function or
			the rational quadratic kernel function.
		
		r2 : float
			The multiplicative coefficient associated with the polynomial kernel function of degree 2.
	
		c : float
			The non-homogenous additive constant in the polynomial kernel function of degree 2.
			
		bw : float
			The bandwidth parameter in the Gaussian kernel function or the rational quadratic kernel function;
			must be strictly positive.
			
		kernel_type : str
			The type of the kernel function used; must be one of 'gaussian_poly2' and 'rationalquad_poly2'.
		
		"""
		
		# construct the contaminated density estimate
		self.contam_density = ContamMLDensityEstimate(
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
		self.uncontam_density = ContamMLDensityEstimate(
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
	
	def eval_IF_logdensity(self, new_data, basis_type, optalgo_params, batchmc_params, grid_points=None,
						   random_seed=0, step_size_factor=1.0, algo='gd', print_error=True):
		
		"""
		Evaluates the influence functions of the logarithm of the maximum likelihood density estimate at new_data.

		Parameters
		----------
		new_data : numpy.ndarray
			An array of data points at which the influence function of the logarithm of
			the maximum likelihood density estimate is to be evaluated.
		
		basis_type : str
			The type of the basis functions of the natural parameter in the density estimate;
			must be one of 'gubasis' and 'grid_points'.
			
		optalgo_params : dict
			The dictionary of parameters to control the gradient descent algorithm.
			Must be returned from the function negloglik_optalgoparams.
		
		batchmc_params : dict
			The dictionary of parameters to control the batch Monte Carlo method
			to approximate the partition function and the gradient of the log-partition function.
			Must be returned from the function batch_montecarlo_params.
			
		grid_points : numpy.ndarray or None, optional
			The set of grid points at which the kernel functions are centered;
			default is None.
		
		random_seed : int, optional
			The seed number to initiate the random number generator; default is 0.
		
		step_size_factor : float, optional
			The multiplicative constant applied to the step size at each iteration; default is 1.
			This constant has the effect that, if step_size_factor is between 0 and 1,
			as the algorithm progresses, the step size is becoming smaller.
			If it is equal to 1, the step size does not change.
		
		algo : str, optional
			The algorithm used to minimize the penalized negative log-likelihood loss function;
			must be one of 'gd', the gradient descent algorithm, or 'newton', the Newton's method;
			default is 'gd'.
		
		print_error : bool, optional
			Whether to print the error of the optimization algorithm at each iteration; default is True.

		Returns
		-------
		numpy.ndarray
			The value of the influence function of the logarithm of
			the maximum likelihood density estimate at new_data.
			
		"""
		
		if self.contam_density.contam_weight == 0.:
			raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
		
		if basis_type not in ['gubasis', 'grid_points']:
			raise ValueError("basis_type must be one of 'gubasis' and 'grid_points'.")
		
		if basis_type == 'grid_points' and grid_points is None:
			raise ValueError("The basis_type is 'grid_points', under which condition grid_points cannot be None. ")
		
		# related to contaminated density
		print('-' * 50)
		print('Computing the coef associated with the contaminated density estimate...')
		if basis_type == 'gubasis':
			
			np.random.seed(random_seed)
			contamde_coef = self.contam_density.coef_gubasis(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				print_error=print_error)
		
		elif basis_type == 'grid_points':
			
			np.random.seed(random_seed)
			contamde_coef = self.contam_density.coef_grid_points(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				step_size_factor=step_size_factor,
				algo=algo,
				grid_points=grid_points,
				print_error=print_error)
		
		contam_results = self.contam_density.log_density(
			new_data=new_data,
			coef=contamde_coef,
			compute_base_density=False)
		
		contam_logden = contam_results
		
		# related to contaminated density
		print('-' * 50)
		print('Computing the coef associated with the uncontaminated density estimate...')
		if basis_type == 'gubasis':
			
			np.random.seed(random_seed)
			uncontamde_coef = self.uncontam_density.coef_gubasis(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				print_error=print_error)
		
		elif basis_type == 'grid_points':
			
			np.random.seed(random_seed)
			uncontamde_coef = self.uncontam_density.coef_grid_points(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				step_size_factor=step_size_factor,
				algo=algo,
				grid_points=grid_points,
				print_error=print_error)
			
		uncontam_results = self.uncontam_density.log_density(
			new_data=new_data,
			coef=uncontamde_coef,
			compute_base_density=False)
		
		uncontam_logden = uncontam_results
		
		# apply the finite difference method to approximate the influence function
		output = (contam_logden - uncontam_logden) / self.contam_density.contam_weight
		
		return output
	
	def plot_IF_logdensity_1d(self, basis_type, optalgo_params, batchmc_params, plot_kwargs, x_label,
							  grid_points=None, step_size_factor=None, random_seed=0, algo='gd',
							  print_error=True, save_plot=False, save_dir=None, save_filename=None):
		
		"""
		Computes and plots the influence function of the logarithm of
		the maximum likelihood density estimate over a specified bounded interval.
		
		Parameters
		----------
		basis_type : str
			The type of the basis functions of the natural parameter in the density estimate;
			must be one of 'gubasis' and 'grid_points'.
			
		optalgo_params : dict
			The dictionary of parameters to control the gradient descent algorithm.
			Must be returned from the function negloglik_optalgoparams.
		
		batchmc_params : dict
			The dictionary of parameters to control the batch Monte Carlo method
			to approximate the partition function and the gradient of the log-partition function.
			Must be returned from the function batch_montecarlo_params.
		
		plot_kwargs : dict
			The dict containing plotting parameters returned from the function plot_IF_1d_params.
		
		x_label : str
			The label of the horizontal axis.
		
		grid_points : numpy.ndarray or None, optional
			The set of grid points at which the kernel functions are centered;
			default is None.
		
		step_size_factor : float, optional
			The multiplicative constant applied to the step size at each iteration; default is 1.
			This constant has the effect that, if step_size_factor is between 0 and 1,
			as the algorithm progresses, the step size is becoming smaller.
			If it is equal to 1, the step size does not change.
		
		random_seed : int, optional
			The seed number to initiate the random number generator; default is 0.
		
		algo : str, optional
			The algorithm used to minimize the penalized negative log-likelihood loss function;
			must be one of 'gd', the gradient descent algorithm, or 'newton', the Newton's method;
			default is 'gd'.
		
		print_error : float, optional
			Whether to print the error of the minimization algorithm at each iteration; default is True.
		
		save_plot : bool, optional
			Whether to save the plot of the influence function to a local file; default is False.

		save_dir : str or None, optional
			The directory path to which the plot of the influence function is saved;
			only works when save_plot is set to be True. Default is None.
	
		save_filename : str or None, optional
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
		result = self.eval_IF_logdensity(
			new_data=new_data,
			basis_type=basis_type,
			optalgo_params=optalgo_params,
			batchmc_params=batchmc_params,
			grid_points=grid_points,
			random_seed=random_seed,
			step_size_factor=step_size_factor,
			algo=algo,
			print_error=print_error)
		
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
	