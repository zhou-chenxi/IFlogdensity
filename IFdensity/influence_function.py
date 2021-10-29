from IFdensity.contam_sm_de import ContamSMDensityEstimate

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
	
	Parameters
	----------
	x_limit:
	y_limit:
	plot_pts_cnt:
	figsize:
	IF_color:
	linewidth:
	rugplot_data_color:
	rugplot_contam_data_color:
	title_fontsize:
	label_fontsize:
	tick_fontsize:
	info_fontsize:
	contam_data_marker_color:
	contam_data_marker_alpha:
	
	
	Returns
	-------
	
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
	contam_density
	
	
	uncontam_density
	
	
	
	base_density
	
	
	
	
	
	Methods
	-------
	eval_IF_logdensity(new_data)
	
	
	
	plot_IF_logdensity_1d(plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None)
	
	
	
	
	
	"""
	
	def __init__(self, data, contam_data, contam_weight, penalty_param, base_density,
				 r1=1.0, r2=0., c=0., bw=1.0, kernel_type='gaussian_poly2'):
		
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
		
		Parameters
		----------
		new_data : numpy.ndarray
	
		Returns
		-------
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

		Parameters
		----------
		new_data : numpy.ndarray

		Returns
		-------
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
	
		pen_param = self.contam_density.penalty_param
		K11 = self.uncontam_density.matrix_K11()
		K13 = self.uncontam_density.matrix_K13()
		N = self.uncontam_density.N
		d = self.uncontam_density.d
		
		K11_inv = np.linalg.inv(K11 + N * pen_param * np.eye(N * d))
		prod1 = np.matmul(K11, K11_inv)
		prod2 = np.matmul(prod1, K13) - 2 * K13
		prod3 = np.matmul(K11_inv, prod2)
		gamma_coef = np.vstack((- pen_param * prod3, - pen_param))
		
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
		
		N, d = self.contam_density.N, self.contam_density.d
		large_K = self.contam_density.matrix_K()
		coef = np.vstack((self.contam_density.coef()[:(N * d + d)],
						  - 1 / self.contam_density.penalty_param,
						  1 / self.contam_density.penalty_param)).reshape(-1, 1)
		output = np.matmul(coef.T, (np.matmul(large_K, coef))).item()
		
		return np.sqrt(output)
	
	def eval_IF_natparam_limit_norm_1d(self):
		
		N, d = self.contam_density.N, self.contam_density.d
		assert d == 1, f'The function eval_IF_natparam_limit_norm_1d only works for 1-dimensional data, ' \
					   f'but got {d}-dimensional data.'
		
		pen_param = self.contam_density.penalty_param
		K11 = self.contam_density.matrix_K11()
		K13 = self.contam_density.matrix_K13()
		K11_inv = np.linalg.inv(K11 + N * pen_param * np.eye(N * d))
		prod1 = np.matmul(K11, K11_inv)
		prod2 = np.matmul(prod1, K13) - 2 * K13
		prod3 = np.matmul(K11_inv, prod2)
		gamma_coef = - pen_param * prod3
		
		part1 = np.matmul(gamma_coef.T, (np.matmul(K11, gamma_coef))).item()
		
		# norm{z_{F_n}}^2 / pen_param ** 2
		part2 = self.contam_density.matrix_K33().item() / pen_param ** 2
		
		# norm{z_{delta_y}}^2 / pen_param ** 2
		part3 =
		
		# inner product between partial_u k (X_i, .) and z_{F_n}
		part4 = np.sum(K13.flatten() * gamma_coef.flatten()) * 2 / pen_param
		
		output = part1 + part2 + part3 - part4
		
		return np.sqrt(output)
	
	
	
	
	
	
	
		
	
	
	
	
	
	def plot_IF_logdensity_1d(self, plot_kwargs, x_label, save_plot=False, save_dir=None, save_filename=None):
		
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
	