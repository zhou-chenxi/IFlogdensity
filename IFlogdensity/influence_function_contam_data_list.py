import os
import numpy as np
from IFlogdensity.contam_sm_de import ContamSMDensityEstimate
from IFlogdensity.contam_ml_de import *

def eval_IF_SMlogdensity_contam_data_array(data, new_data, contam_data_array, contam_weight,
										penalty_param, base_density,
										r1=1.0, r2=0., c=0., bw=1.0, kernel_type='gaussian_poly2',
										save_data=False, save_dir=None):
	
	"""
	Evaluates the influence function of the logarithm of the score matching density estimate at new_data.
	The result is a dict, where each key corresponds to a distinct contaminated observation in contam_data_array.
	
	Parameters
	----------
	data : numpy.ndarray
		The array of observations whose probability density function is to be estimated.
	
	new_data : numpy.ndarray
		The array of data points at which the influence function of the logarithm of
		the score matching density estimate is to be evaluated.
		
	contam_data_array : numpy.ndarray
		The array of contaminated observations.
		
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

	save_data : bool, optional
		Whether or not to save the values of the influence function of
		the logarithm of the score matching density estimate as a local file; default is False.
	
	save_dir : str or None, optional
		The directory path to which the values of the influence function of
		the logarithm of the score matching density estimate is saved;
		only works when save_plot is set to be True. Default is None.
	
	Returns
	-------
	dict
		A dict of the values of the influence function of the the logarithm of
		the score matching density estimate at new_data,
		where each key corresponds to a distinct contaminated observation in contam_data_array.
	
	"""
	
	if contam_weight == 0.:
		raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
	
	# check the validity of the contam_data_array
	if not isinstance(contam_data_array, np.ndarray):
		raise TypeError(f'contam_data_array must be a numpy.ndarray, but got {type(contam_data_array)}.')
	
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
	
	# check the compatibility of data and contam_data_array
	if len(contam_data_array.shape) == 1:
		contam_data_array = contam_data_array.reshape(-1, 1)
	if contam_data_array.shape[1] != d:
		raise ValueError('contam_data_array are not compatible with data and new_data.')
	
	print('-' * 50)
	print('Computing the uncontaminated log-density values.')
	# compute the log-density values of the uncontaminated data
	uncontam_den = ContamSMDensityEstimate(
		data=data,
		contam_data=contam_data_array[0].reshape(1, d),
		contam_weight=0.,
		penalty_param=penalty_param,
		base_density=base_density,
		r1=r1,
		r2=r2,
		c=c,
		bw=bw,
		kernel_type=kernel_type)
	
	uncontam_logdenvals_new = uncontam_den.log_density(new_data=new_data)
	uncontam_logdenvals_contam = uncontam_den.log_density(new_data=contam_data_array)
	
	# save data
	if save_data:
		full_save_folder = 'data/' + save_dir
		if not os.path.isdir(full_save_folder):
			os.mkdir(full_save_folder)
		
		file_name_newdata = f'/new_data.npy'
		np.save(full_save_folder + file_name_newdata, new_data)
		
		file_name_diff = f'/uncontam-logden-newdata.npy'
		np.save(full_save_folder + file_name_diff, uncontam_logdenvals_new)
		
		file_name_contamdata = f'/contam_data.npy'
		np.save(full_save_folder + file_name_contamdata, contam_data_array)
		
		file_name_logden_contam = f'/uncontam-logden-contamdata.npy'
		np.save(full_save_folder + file_name_logden_contam, uncontam_logdenvals_contam)
	
	IF_output_new = {}
	IF_output_new['new_data'] = new_data
	IF_output_new['contam_data'] = contam_data_array
	
	IF_output_contam = {}
	
	for i in range(len(contam_data_array)):
		
		print('-' * 50)
		print(f'Computing the contaminated log-density values ')
		print(f'with the current contaminated data point being {contam_data_array[i]}.')
		
		contam_den = ContamSMDensityEstimate(
			data=data,
			contam_data=contam_data_array[i].reshape(1, d),
			contam_weight=contam_weight,
			penalty_param=penalty_param,
			base_density=base_density,
			r1=r1,
			r2=r2,
			c=c,
			bw=bw,
			kernel_type=kernel_type)
		
		contam_logdenvals_new = contam_den.log_density(new_data=new_data)
		contam_logdenvals_contam = contam_den.log_density(new_data=contam_data_array)
		
		IF_newdata = (contam_logdenvals_new - uncontam_logdenvals_new) / contam_weight
		IF_output_new['contam ' + str(contam_data_array[i])] = IF_newdata
		
		IF_contam = (contam_logdenvals_contam - uncontam_logdenvals_contam) / contam_weight
		IF_output_contam['contam ' + str(contam_data_array[i])] = IF_contam
		
		if save_data:
			file_name_diff = f'/contam_data={contam_data_array[i]}-contam-logden-newdata.npy'
			np.save(full_save_folder + file_name_diff, contam_logdenvals_new)
			
			file_name_logden_contam = f'/contam_data={contam_data_array[i]}-contam-logden-contamdata.npy'
			np.save(full_save_folder + file_name_logden_contam, contam_logdenvals_contam)
			
			IF_file_name_new = f'/contam_data={contam_data_array[i]}-IF-logden-newdata.npy'
			np.save(full_save_folder + IF_file_name_new, IF_newdata)
			
			IF_file_name_contam = f'/contam_data={contam_data_array[i]}-IF-logden-contamdata.npy'
			np.save(full_save_folder + IF_file_name_contam, IF_contam)
	
	# form the final output of influence function at contam_data_list
	IF_contam_list = []
	for i in range(len(contam_data_array)):
		IF_contam_list.append(IF_output_contam['contam ' + str(contam_data_array[i])])
	
	IF_contam_diag = np.diag(np.array(IF_contam_list))
	IF_contam_final_output = {'contam_data': contam_data_array, 'IF_vals': IF_contam_diag}
	
	return IF_output_new, IF_contam_final_output


def eval_IF_MLlogdensity_contam_data_array(data, new_data, contam_data_array, contam_weight,
										penalty_param, base_density, basis_type,
										optalgo_params, batchmc_params,
										r1=1.0, r2=0., c=0., bw=1.0, kernel_type='gaussian_poly2',
										grid_points=None, algo='gd', step_size_factor=1.0,
										random_seed=0, save_data=False, save_dir=None, print_error=True):
	
	"""
	Evaluates the influence function of the logarithm of the maximum likelihood density estimate at new_data.
	The result is a dict, where each key corresponds to a distinct contaminated observation in contam_data_array.
	
	Parameters
	----------
	data : numpy.ndarray
		The array of observations whose probability density function is to be estimated.
	
	new_data : numpy.ndarray
		The array of data points at which the influence function of the logarithm of
		the maximum likelihood density estimate is to be evaluated.
		
	contam_data_array : numpy.ndarray
		The array of contaminated observations.
		
	grid_points : numpy.ndarray
		The array of grid points at which the basis functions of the natural parameter are centered.

	contam_weight : float
		The weight of contamination.

	penalty_param : float
		The penalty parameter. Must be strictly positive.

	base_density : base_density object
		The base density function used to estimate the probability density function.

	basis_type : str
		The type of the basis functions in the natural parameter;
		must be one of 'gubasis' and 'grid_points'.
		
	optalgo_params : dict
		The dictionary of parameters to control the gradient descent algorithm.
		Must be returned from the function negloglik_optalgoparams.
	
	batchmc_params : dict
		The dictionary of parameters to control the batch Monte Carlo method
		to approximate the partition function and the gradient of the log-partition function.
		Must be returned from the function batch_montecarlo_params.
		
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

	grid_points : numpy.ndarray or None, optional
		The set of grid points at which the kernel functions are centered;
		default is None.

	algo : str, optional
		The algorithm used to minimize the penalized negative log-likelihood loss function;
		must be one of 'gd', the gradient descent algorithm, or 'newton', the Newton's method;
		default is 'gd'.
	
	step_size_factor : float, optional
		The multiplicative constant applied to the step size at each iteration; default is 1.
		This constant has the effect that, if step_size_factor is between 0 and 1,
		as the algorithm progresses, the step size is becoming smaller.
		If it is equal to 1, the step size does not change.

	random_seed : int, optional
		The seed number to initiate the random number generator; default is 0.

	save_data : bool, optional
		Whether or not to save the values of the influence function of
		the logarithm of the maximum likelihood density estimate as a local file; default is False.
	
	save_dir : str or None, optional
		The directory path to which the values of the influence function of
		the logarithm of the maximum likelihood density estimate is saved;
		only works when save_plot is set to be True. Default is None.
	
	print_error : bool, optional
		Whether to print the error of the optimization algorithm at each iteration; default is True.
	
	Returns
	-------
	dict
		A dict of the values of the influence function of the the logarithm of
		the maximum likelihood density estimate at new_data,
		where each key corresponds to a distinct contaminated observation in contam_data_array.
	
	"""
	
	if contam_weight == 0.:
		raise ValueError('In order to compute the influence function, contam_weight cannot be 0.')
	
	# check the validity of the contam_data_array
	if not isinstance(contam_data_array, np.ndarray):
		raise TypeError(f'contam_data_array must be a numpy.ndarray, but got {type(contam_data_array)}.')
	
	# check the compatibility of data and new_data
	if not isinstance(data, np.ndarray):
		data = np.array(data)
	
	if not isinstance(new_data, np.ndarray):
		new_data = np.array(new_data)
	
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	
	if len(new_data.shape) == 1:
		new_data = new_data.reshape(-1, 1)
	
	if basis_type not in ['gubasis', 'grid_points']:
		raise ValueError("The basis_type must be one of 'gubasis' and 'grid_points'.")
	
	if basis_type == 'grid_points' and grid_points is None:
		raise ValueError("The basis_type is 'grid_points', under which condition grid_points cannot be None. ")
	
	N, d = data.shape
	n, d1 = new_data.shape
	if d != d1:
		raise ValueError('data and new_data are not compatible.')
	
	# check the compatibility of data and contam_data_array
	if len(contam_data_array.shape) == 1:
		contam_data_array = contam_data_array.reshape(-1, 1)
	if contam_data_array.shape[1] != d:
		raise ValueError('contam_data_array are not compatible with data and new_data.')
	
	print('-' * 50)
	print('Computing the uncontaminated log-density values.')
	# compute the log-density values of the uncontaminated data
	uncontam_den = ContamMLDensityEstimate(
		data=data,
		contam_data=contam_data_array[0].reshape(1, d),
		contam_weight=0.,
		penalty_param=penalty_param,
		base_density=base_density,
		r1=r1,
		r2=r2,
		c=c,
		bw=bw,
		kernel_type=kernel_type)
	
	if basis_type == 'gubasis':

		np.random.seed(random_seed)
		uncontam_coef = uncontam_den.coef_gubasis(
			optalgo_params=optalgo_params,
			batchmc_params=batchmc_params,
			print_error=print_error)
	
	elif basis_type == 'grid_points':
		
		np.random.seed(random_seed)
		uncontam_coef = uncontam_den.coef_grid_points(
			optalgo_params=optalgo_params,
			batchmc_params=batchmc_params,
			step_size_factor=step_size_factor,
			algo=algo,
			grid_points=grid_points,
			print_error=print_error)
	
	uncontam_logdenvals_new = uncontam_den.log_density(
		new_data=new_data,
		coef=uncontam_coef,
		compute_base_density=False)
	
	# save data
	if save_data:
		full_save_folder = 'data/' + save_dir
		if not os.path.isdir(full_save_folder):
			os.mkdir(full_save_folder)
		
		file_name_newdata = f'/new_data.npy'
		np.save(full_save_folder + file_name_newdata, new_data)
		
		print('new_data saved.')
		
		file_name_grid_points = f'/grid_points.npy'
		np.save(full_save_folder + file_name_grid_points, grid_points)
		
		print('grid_points saved.')
		
		file_name_contamdata = f'/contam_data.npy'
		np.save(full_save_folder + file_name_contamdata, contam_data_array)
		
		print('contam_data_array saved.')
		
		file_name_coef = f'/uncontam-coef.npy'
		np.save(full_save_folder + file_name_coef, uncontam_coef[0])
		
		print('uncontam_coef saved.')
		
		file_name_diff = f'/uncontam-logden-newdata.npy'
		np.save(full_save_folder + file_name_diff, uncontam_logdenvals_new)
		print('uncontam_logdensity saved.')
	
	IF_output_new = {}
	IF_output_new['new_data'] = new_data
	
	for i in range(len(contam_data_array)):
		
		print('-' * 50)
		print(f'Computing the contaminated log-density values ')
		print(f'with the current contaminated data point being {contam_data_array[i]}.')
		
		contam_den = ContamMLDensityEstimate(
			data=data,
			contam_data=contam_data_array[i].reshape(1, d),
			contam_weight=contam_weight,
			penalty_param=penalty_param,
			base_density=base_density,
			r1=r1,
			r2=r2,
			c=c,
			bw=bw,
			kernel_type=kernel_type)
		
		if basis_type == 'gubasis':
			
			np.random.seed(random_seed)
			contam_coef = contam_den.coef_gubasis(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				print_error=print_error)
		
		elif basis_type == 'grid_points':
			
			np.random.seed(random_seed)
			contam_coef = contam_den.coef_grid_points(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				step_size_factor=step_size_factor,
				algo=algo,
				grid_points=grid_points,
				print_error=print_error)
			
		contam_logdenvals_new = contam_den.log_density(
			new_data=new_data,
			coef=contam_coef,
			compute_base_density=False)
		
		IF_newdata = (contam_logdenvals_new - uncontam_logdenvals_new) / contam_weight
		IF_output_new['contam ' + str(contam_data_array[i])] = IF_newdata
		
		if save_data:
			# save the coef
			IF_file_name_coef = f'/contam_data={contam_data_array[i]}-contam-coef.npy'
			np.save(full_save_folder + IF_file_name_coef, contam_coef[0])
			
			# save the log density values
			file_name_diff = f'/contam_data={contam_data_array[i]}-contam-logden-newdata.npy'
			np.save(full_save_folder + file_name_diff, contam_logdenvals_new)
			
			# save the IF
			IF_file_name_new = f'/contam_data={contam_data_array[i]}-IF-logden-newdata.npy'
			np.save(full_save_folder + IF_file_name_new, IF_newdata)
	
	return IF_output_new
