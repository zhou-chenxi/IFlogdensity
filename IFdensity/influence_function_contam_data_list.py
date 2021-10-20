import os
import numpy as np
from IFdensity.contam_sm_de import ContamSMDensityEstimate


def eval_IF_logdensity_contam_data_list(data, new_data, contam_data_list, contam_weight, penalty_param, base_density,
										r1=1.0, r2=0., c=0., bw=1.0, kernel_type='gaussian_poly2',
										save_data=False, save_dir=None):
	
	if contam_weight == 0.:
		raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
	
	# check the validity of the contam_data_list
	if not isinstance(contam_data_list, list):
		
		raise TypeError(f'contam_data_list must be a list, but got {type(contam_data_list)}.')
	
	# check the compatibility of data and new_data
	if not isinstance(data, np.ndarray):
		data = np.array(data)
	
	if not isinstance(new_data, np.ndarray):
		new_data = np.array(new_data)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	if len(new_data.shape) == 1:
		new_data = new_data.reshape(-1, 1)
		
	N, d = data.shape
	n, d1 = new_data.shape
	if d != d1:
		raise ValueError('data and new_data are not compatible.')
	
	# check the compatibility of data and contam_data_list
	contam_data_list_len = np.array([len(item) for item in contam_data_list])
	if len(np.unique(contam_data_list_len)) != 1:
		raise ValueError('The elements in contam_data_list are not of the same length.')
	if np.unique(contam_data_list_len) != d:
		raise ValueError('contam_data_list are not compatible with data and new_data.')
	
	# compute the log-density values of the uncontaminated data
	uncontam_den = ContamSMDensityEstimate(
		data=data,
		contam_data=contam_data_list[0],
		contam_weight=contam_weight,
		penalty_param=penalty_param,
		base_density=base_density,
		r1=r1,
		r2=r2,
		c=c,
		bw=bw,
		kernel_type=kernel_type)
	
	uncontam_logdenvals_new = uncontam_den.log_density(new_data=new_data)['logden_vals']
	uncontam_logdenvals_contam = uncontam_den.log_density(new_data=np.array(contam_data_list).reshape(
		len(contam_data_list), data.shape[1]))['logden_vals']
	
	# save data
	if save_data:
		full_save_folder = 'data/' + save_dir
		if not os.path.isdir(full_save_folder):
			os.path.mkdir(full_save_folder)
		
		file_name_newdata = f'/new_data.npy'
		np.save(full_save_folder + file_name_newdata, new_data)
		
		file_name_diff = f'/bw={bw}-kernel={kernel_type}-lambda-{penalty_param}-uncontam-logden-newdata.npy'
		np.save(full_save_folder + file_name_diff, uncontam_logdenvals_new)
		
		file_name_contamdata = f'/contam_data.npy'
		np.save(full_save_folder + file_name_contamdata, np.array(contam_data_list).reshape(
			len(contam_data_list), data.shape[1]))
		
		file_name_logden_contam = f'/bw={bw}-kernel={kernel_type}-lambda-{penalty_param}-uncontam-logden-contamdata.npy'
		np.save(full_save_folder + file_name_logden_contam, uncontam_logdenvals_contam)
	
	IF_output_new = {}
	IF_output_contam = {}
	
	for contam_data in contam_data_list:
		
		print('-' * 50)
		print(f'Current contaminated data point is {contam_data}.')
		
		contam_den = ContamSMDensityEstimate(
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
		
		contam_logdenvals_new = contam_den.log_density(new_data=new_data)
		contam_logdenvals_contam = contam_den.log_density(new_data=np.array(contam_data_list).reshape(
			len(contam_data_list), data.shape[1]))
		
		IF_output_new['contam ' + str(contam_data)] = (contam_logdenvals_new - uncontam_logdenvals_new) / contam_weight
		IF_output_contam['contam ' + str(contam_data)] = ((contam_logdenvals_contam - uncontam_logdenvals_contam) /
														  contam_weight)
		
		if save_data:
			
			file_name_diff = f'/bw={bw}-kernel={kernel_type}-lambda-{penalty_param}-contam-logden-newdata.npy'
			np.save(full_save_folder + file_name_diff, contam_logdenvals_new)
			
			file_name_logden_contam = f'/bw={bw}-kernel={kernel_type}-lambda-{penalty_param}-contam-logden-contamdata.npy'
			np.save(full_save_folder + file_name_logden_contam, contam_logdenvals_contam)
		
	return IF_output_new, IF_output_contam
