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
	
	def eval_IF_logdensity(self, new_data):
		
		if self.contam_density.contam_weight == 0.:
			raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
		
		# contaminated log-density function part
		contam_coef = self.contam_density.coef()
		contam_natparam = self.contam_density.natural_param(
			new_data=new_data,
			coef=contam_coef)
		contam_logpar = self.contam_density.density_logpartition_1d(coef=contam_coef)
		
		# uncontaminated log-density function part
		uncontam_coef = self.uncontam_density.coef()
		uncontam_natparam = self.uncontam_density.natural_param(
			new_data=new_data,
			coef=uncontam_coef)
		uncontam_logpar = self.uncontam_density.density_logpartition_1d(coef=uncontam_coef)
		
		# apply the finite difference method to approximate the influence function
		output = ((contam_natparam - contam_logpar) -
				  (uncontam_natparam - uncontam_logpar)) / self.contam_density.contam_weight
		
		return output
	
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
