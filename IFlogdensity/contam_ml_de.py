import numpy as np
from dekef.check import *
from dekef.base_density import *
from dekef.kernel_function import *
from scipy import integrate


def batch_montecarlo_params(mc_batch_size=1000, mc_tol=1e-2):
	"""
	Returns a dictionary of parameters for the batch Monte Carlo method
	in approximating the log-partition function and its gradient.
	Parameters
	----------
	mc_batch_size : int
		The batch size in the batch Monte Carlo method; default is 1000.
	mc_tol : float
		The floating point number below which sampling in the batch Monte Carlo is terminated; default is 1e-2.
	Returns
	-------
	dict
		The dictionary containing both supplied parameters.
	"""
	
	mc_batch_size = int(mc_batch_size)
	
	output = {"mc_batch_size": mc_batch_size,
			  "mc_tol": mc_tol}
	
	return output


def negloglik_optalgoparams(start_pt, step_size=0.01, max_iter=1e2, rel_tol=1e-5):
	if not isinstance(max_iter, int):
		max_iter = int(max_iter)
	
	assert step_size > 0., 'step_size must be strictly positive.'
	assert rel_tol > 0., 'rel_tol must be strictly positive.'
	
	output = {"start_pt": start_pt,
			  "step_size": step_size,
			  "max_iter": max_iter,
			  "rel_tol": rel_tol}
	
	return output


def finer_grid_points(grid_points):
	if len(grid_points.shape) == 2 and grid_points.shape[1] != 1:
		raise ValueError((f'The function finer_grid_points only works for the 1-dimensional grid points, '
						  f'but got {grid_points.shape[1]}-dimensional.'))
	
	grid_points = np.sort(grid_points.flatten())
	
	double_gp = np.array([grid_points[1:].flatten(), grid_points[:(len(grid_points) - 1)].flatten()])
	new_gp = np.mean(double_gp, axis=0)
	output = np.sort(np.concatenate([new_gp, grid_points]))
	
	return output


class ContamMLDensityEstimate:
	"""
	A class of estimating the probability density function via maximizing the log-likelihood function
	in the presence of a contaminated observation.

	...

	Attributes
	----------




	Methods
	-------

	coef()
		Computes the coefficient vector of the natural parameter in the score matching density estimate.

	natural_param(new_data, coef)
		Evaluates the natural parameter in the score matching density estimate at new_data.

	unnormalized_density_eval_1d(x, coef)
		Evaluates the un-normalized score matching density estimate at new_data.

	density_logpartition_1d(coef)
		Compute the normalizing constant of the score matching density estimate.

	log_density(new_data, compute_base_density)
		Compute the logarithm of the score matching density estimate.

	"""
	
	def __init__(self, data, contam_data, contam_weight, penalty_param, base_density,
				 r1=1.0, r2=0.0, c=0., bw=1., kernel_type='gaussian_poly2'):
		
		# check types of data and contam_data
		if isinstance(data, np.ndarray):
			data = np.array(data)
		
		if isinstance(contam_data, np.ndarray):
			contam_data = np.array(contam_data)
		
		# check compatibility of data and contam_data
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		if len(contam_data.shape) == 1:
			contam_data = contam_data.reshape(-1, 1)
		
		N, d = data.shape
		n, d1 = contam_data.shape
		
		if d != d1:
			raise ValueError('The shape of data and contam_data are not compatible.')
		
		if n != 1:
			raise ValueError('There are multiple contaminated data. Please just supply one.')
		
		# check the validity of banwidth parameter
		if bw <= 0.:
			raise ValueError("The bw parameter must be strictly positive.")
		
		# check the validity of kernel type
		if kernel_type not in ['gaussian_poly2', 'rationalquad_poly2']:
			raise ValueError(f"The kernle_type must be one of 'gaussian_poly2' and 'rationalquad_poly2', "
							 f"but got {kernel_type}.")
		
		self.data = data
		self.contam_data = contam_data
		self.N = N
		self.n = n
		self.d = d
		
		# check the validity of the contam_weight
		assert 0. <= contam_weight <= 1., "contam_weight must be between 0 and 1, inclusively."
		self.contam_weight = contam_weight
		
		# check the validity of the penalty_param
		assert penalty_param > 0., "penalty_param must be strictly positive, inclusively."
		self.penalty_param = penalty_param
		
		# check the base density
		check_basedensity(base_density)
		self.base_density = base_density
		
		# construct kernel function
		self.r1 = r1
		self.r2 = r2
		self.c = c
		self.bw = bw
		
		if kernel_type == 'gaussian_poly2':
			
			self.kernel_type = 'gaussian_poly2'
			
			self.kernel_function_data = GaussianPoly2(
				data=data,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
			
			self.kernel_function_contam_data = GaussianPoly2(
				data=contam_data,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
		
		elif kernel_type == 'rationalquad_poly2':
			
			self.kernel_type = 'rationalquad_poly2'
			
			self.kernel_function_data = RationalQuadPoly2(
				data=data,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
			
			self.kernel_function_contam_data = RationalQuadPoly2(
				data=contam_data,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
		
		else:
			
			raise NotImplementedError(f"kernel_type must be one of 'gaussian_poly2' and 'rationalquad_poly2', "
									  f"but got {kernel_type}.")
	
	def grad_logpar_batchmc(self, coef, basis_type, grid_points=None, mc_batch_size=1000,
							mc_tol_param=1e-2, normalizing_const_only=False, print_error=False):
		
		if len(self.data.shape) == 1:
			self.data = self.data.reshape(-1, 1)
		
		if len(coef.shape) == 1:
			coef = coef.reshape(-1, 1)
		
		if not isinstance(mc_batch_size, int):
			mc_batch_size = int(mc_batch_size)
		
		if basis_type == 'gubasis':
			
			# gubasis means the basis functions are the kernel basis functions
			# with one argument being the data point
			n_basis = self.N + self.n
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function = GaussianPoly2(
					data=np.vstack((self.data, self.contam_data)),
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function = RationalQuadPoly2(
					data=np.vstack((self.data, self.contam_data)),
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
		
		elif basis_type == 'grid_points':
			
			if grid_points is None:
				raise ValueError('To use grid_points, please supply grid_points.')
			
			if len(grid_points.shape) == 1:
				
				grid_points = grid_points.reshape(-1, 1)
			
			elif grid_points.shape[1] != self.d:
				
				raise ValueError(f'The dimensionality of grid_points does not match that of the data.')
			
			n_basis = grid_points.shape[0]
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function = GaussianPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function = RationalQuadPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
		
		else:
			
			raise NotImplementedError("basis_type must be one of 'gubasis' and 'grid_points'.")
		
		assert len(coef) == n_basis, 'The length of coef does not match as expected. Please double check.'
		
		###########################################################################
		# estimate the normalizing constant
		# first drawing
		mc_samples1 = self.base_density.sample(mc_batch_size)
		mc_kernel_matrix1 = kernel_function.kernel_gram_matrix(mc_samples1)
		unnorm_density_part1 = np.exp(np.matmul(mc_kernel_matrix1.T, coef))
		norm_const1 = np.mean(unnorm_density_part1)
		
		# second drawing
		mc_samples2 = self.base_density.sample(mc_batch_size)
		mc_kernel_matrix2 = kernel_function.kernel_gram_matrix(mc_samples2)
		unnorm_density_part2 = np.exp(np.matmul(mc_kernel_matrix2.T, coef))
		norm_const2 = np.mean(unnorm_density_part2)
		
		norm_est_old = norm_const1
		norm_est_new = (norm_const1 + norm_const2) / 2
		
		error_norm = np.abs(norm_est_old - norm_est_new) / norm_est_old
		
		if print_error:
			print('normalizing constant error = {error:.7f}'.format(error=error_norm))
		
		batch_cnt = 2
		
		while error_norm > mc_tol_param:
			
			norm_est_old = norm_est_new
			
			# another draw
			mc_samples = self.base_density.sample(mc_batch_size)
			mc_kernel_matrix = kernel_function.kernel_gram_matrix(mc_samples)
			unnorm_density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef))
			norm_const2 = np.mean(unnorm_density_part)
			
			# update the Monte Carlo estimation
			norm_est_new = (norm_est_old * batch_cnt + norm_const2) / (batch_cnt + 1)
			
			batch_cnt += 1
			
			error_norm = np.abs(norm_est_old - norm_est_new) / norm_est_old
			
			if print_error:
				print('normalizing constant error = {error:.7f}'.format(error=error_norm))
		
		normalizing_const = norm_est_new
		
		if not normalizing_const_only:
			
			# estimating the log-partition function
			
			if print_error:
				print("#" * 45 + "\nEstimating the gradient of the log-partition now.")
			
			mc_samples1 = self.base_density.sample(mc_batch_size)
			mc_kernel_matrix1 = kernel_function.kernel_gram_matrix(mc_samples1)
			density_part1 = np.exp(np.matmul(mc_kernel_matrix1.T, coef).flatten()) / normalizing_const
			exp_est1 = np.array([np.mean(mc_kernel_matrix1[l1, :] * density_part1)
								 for l1 in range(n_basis)]).astype(np.float64).reshape(1, -1)[0]
			
			mc_samples2 = self.base_density.sample(mc_batch_size)
			mc_kernel_matrix2 = kernel_function.kernel_gram_matrix(mc_samples2)
			density_part2 = np.exp(np.matmul(mc_kernel_matrix2.T, coef).flatten()) / normalizing_const
			exp_est2 = np.array([np.mean(mc_kernel_matrix2[l1, :] * density_part2)
								 for l1 in range(n_basis)]).astype(np.float64).reshape(1, -1)[0]
			
			grad_est_old = exp_est1
			grad_est_new = (exp_est1 + exp_est2) / 2
			
			error_grad = np.linalg.norm(grad_est_old - grad_est_new, 2) / (np.linalg.norm(grad_est_old, 2) * n_basis)
			
			if print_error:
				print('gradient error = {error:.7f}'.format(error=error_grad))
			
			batch_cnt = 2
			
			while error_grad > mc_tol_param:
				
				grad_est_old = grad_est_new
				
				# another draw
				mc_samples = self.base_density.sample(mc_batch_size)
				mc_kernel_matrix = kernel_function.kernel_gram_matrix(mc_samples)
				density_part = np.exp(np.matmul(mc_kernel_matrix.T, coef).flatten()) / normalizing_const
				exp_est2 = np.array([np.mean(mc_kernel_matrix[l1, :] * density_part)
									 for l1 in range(n_basis)]).astype(np.float64).reshape(1, -1)[0]
				
				grad_est_new = (grad_est_old * batch_cnt + exp_est2) / (batch_cnt + 1)
				
				batch_cnt += 1
				
				error_grad = np.linalg.norm(grad_est_old - grad_est_new, 2) / (
							np.linalg.norm(grad_est_old, 2) * n_basis)
				
				if print_error:
					print('gradient error = {error:.7f}'.format(error=error_grad))
		
		if normalizing_const_only:
			return norm_est_new
		else:
			return norm_est_new, grad_est_new
	
	def coef(self, basis_type, optalgo_params, batchmc_params, grid_points=None, step_size_discount_factor=0.5,
			 max_set_grid_points=5, rel_tol_param=1e-3, print_error=True):
		
		if basis_type == 'gubasis':
			
			result = self.coef_gubasis(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				print_error=print_error)
			
			output = (result, 'gubasis')
		
		elif basis_type == 'grid_points':
			
			result = self.coef_grid_points(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				grid_points=grid_points,
				print_error=print_error)
			
			output = (result, grid_points, 'grid_points')
		
		elif basis_type == 'sieve':
			
			result, final_grid_points = self.coef_sieve(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				start_grid_points=grid_points,
				step_size_discount_factor=step_size_discount_factor,
				max_set_grid_points=max_set_grid_points,
				rel_tol_param=rel_tol_param,
				print_error=print_error)
			
			output = (result, final_grid_points, 'sieve')
		
		return output
	
	def coef_gubasis(self, optalgo_params, batchmc_params, print_error=True):
		
		# parameters associated with gradient descent algorithm
		start_pt = optalgo_params["start_pt"]
		step_size = optalgo_params["step_size"]
		max_iter = optalgo_params["max_iter"]
		rel_tol = optalgo_params["rel_tol"]
		
		if len(start_pt) != self.N + self.n:
			raise ValueError(("The supplied start_pt in optalgo_params is not correct. "
							  "The length of start_pt is expected to be {exp_len}, but got {act_len}.").format(
				exp_len=self.N + self.n, act_len=len(start_pt)))
		
		# parameters associated with batch Monte Carlo estimation
		mc_batch_size = batchmc_params["mc_batch_size"]
		mc_tol = batchmc_params["mc_tol"]
		
		# the gradient of the loss function is
		# nabla L (alpha) = nabla A (alpha) - (1 / n) gram_matrix boldone_n + lambda_param * gram_matrix * alpha
		# the gradient descent update is
		# new_iter = current_iter - step_size * nabla L (alpha)
		
		if self.kernel_type == 'gaussian_poly2':
			
			kernel_function = GaussianPoly2(
				data=np.vstack((self.data, self.contam_data)),
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
		
		elif self.kernel_type == 'rationalquad_poly2':
			
			kernel_function = RationalQuadPoly2(
				data=np.vstack((self.data, self.contam_data)),
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
		
		# form the Gram matrix
		gram_data = kernel_function.kernel_gram_matrix(self.data)
		gram_contamdata = kernel_function.kernel_gram_matrix(self.contam_data)
		grad_f = ((1. - self.contam_weight) * gram_data.mean(axis=1, keepdims=True) +
				  self.contam_weight * gram_contamdata)
		
		gram_all_basis = kernel_function.kernel_gram_matrix(np.vstack((self.data, self.contam_data)))
		
		current_iter = start_pt.reshape(-1, 1)
		
		# compute the gradient of the log-partition function at current_iter
		mc_output1, mc_output2 = self.grad_logpar_batchmc(
			coef=current_iter,
			basis_type='gubasis',
			grid_points=None,
			mc_batch_size=mc_batch_size,
			mc_tol_param=mc_tol,
			normalizing_const_only=False,
			print_error=False)
		
		grad_logpar = mc_output2.reshape(-1, 1)
		
		# compute the gradient of the loss function at current_iter
		current_grad = grad_logpar - grad_f + self.penalty_param * np.matmul(gram_all_basis, current_iter)
		
		# compute the updated iter
		new_iter = current_iter - step_size * current_grad
		
		# compute the error of the first update
		grad0_norm = np.linalg.norm(current_grad, 2)
		error = grad0_norm / (grad0_norm + 1e-8)
		# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
		
		iter_num = 1
		
		if print_error:
			print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
				iter_num=iter_num, gradnorm=grad0_norm, error=error))
		
		while error > rel_tol and iter_num < max_iter:
			
			current_iter = new_iter
			
			# compute the gradient at current_iter
			mc_output1, mc_output2 = self.grad_logpar_batchmc(
				coef=current_iter,
				basis_type='gubasis',
				grid_points=None,
				mc_batch_size=mc_batch_size,
				mc_tol_param=mc_tol,
				normalizing_const_only=False,
				print_error=False)
			
			grad_logpar = mc_output2.reshape(-1, 1)
			
			# compute the gradient of the loss function
			current_grad = grad_logpar - grad_f + self.penalty_param * np.matmul(gram_all_basis, current_iter)
			
			# compute the updated iter
			new_iter = current_iter - step_size * current_grad
			
			# compute the error of the first update
			grad_new_norm = np.linalg.norm(current_grad, 2)
			error = grad_new_norm / (grad0_norm + 1e-8)
			# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
			
			iter_num += 1
			
			if print_error:
				print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
					iter_num=iter_num, gradnorm=grad_new_norm, error=error))
		
		coefficients = new_iter
		
		return coefficients
	
	def coef_grid_points(self, optalgo_params, batchmc_params, grid_points=None, print_error=True):
		
		"""
		Compute the coeffcient vector with a single set of grid points.



		"""
		
		if len(grid_points.shape) == 1:
			grid_points = grid_points.reshape(-1, 1)
		
		assert self.data.shape[1] == grid_points.shape[
			1], 'The dimensionality of data does not match that of grid_points.'
		
		if self.kernel_type == 'gaussian_poly2':
			
			kernel_function_grid = GaussianPoly2(
				data=grid_points,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
		
		elif self.kernel_type == 'rationalquad_poly2':
			
			kernel_function_grid = RationalQuadPoly2(
				data=grid_points,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
		
		# parameters associated with gradient descent algorithm
		start_pt = optalgo_params["start_pt"]
		step_size = optalgo_params["step_size"]
		max_iter = optalgo_params["max_iter"]
		rel_tol = optalgo_params["rel_tol"]
		
		if len(start_pt) != grid_points.shape[0]:
			raise ValueError(("The supplied start_pt in optalgo_params is not correct. "
							  "The expected length of start_pt is {exp_len}, but got {act_len}.").format(
				exp_len=grid_points.shape[0], act_len=len(start_pt)))
		
		# parameters associated with batch Monte Carlo estimation
		mc_batch_size = batchmc_params["mc_batch_size"]
		mc_tol = batchmc_params["mc_tol"]
		
		# the gradient of the loss function is
		# nabla L (alpha) = nabla A (alpha) - (1 / n) gram_matrix boldone_n + lambda_param * gram_matrix * alpha
		# the gradient descent update is
		# new_iter = current_iter - step_size * nabla L (alpha)
		
		# form the Gram matrix
		gram_data = kernel_function_grid.kernel_gram_matrix(self.data)
		gram_contamdata = kernel_function_grid.kernel_gram_matrix(self.contam_data)
		grad_f = ((1. - self.contam_weight) * gram_data.mean(axis=1, keepdims=True) +
				  self.contam_weight * gram_contamdata)
		
		gram_grid = kernel_function_grid.kernel_gram_matrix(grid_points)
		
		current_iter = start_pt.reshape(-1, 1)
		
		# compute the gradient of the log-partition function at current_iter
		mc_output1, mc_output2 = self.grad_logpar_batchmc(
			coef=current_iter,
			basis_type='grid_points',
			grid_points=grid_points,
			mc_batch_size=mc_batch_size,
			mc_tol_param=mc_tol,
			normalizing_const_only=False,
			print_error=False)
		
		grad_logpar = mc_output2.reshape(-1, 1)
		
		# compute the gradient of the loss function at current_iter
		current_grad = grad_logpar - grad_f + self.penalty_param * np.matmul(gram_grid, current_iter)
		
		# compute the updated iter
		new_iter = current_iter - step_size * current_grad
		
		# compute the error of the first update
		grad0_norm = np.linalg.norm(current_grad, 2)
		error = grad0_norm / (grad0_norm + 1e-8)
		# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
		
		iter_num = 1
		
		if print_error:
			print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
				iter_num=iter_num, gradnorm=grad0_norm, error=error))
		
		while error > rel_tol and iter_num < max_iter:
			
			current_iter = new_iter
			
			mc_output1, mc_output2 = self.grad_logpar_batchmc(
				coef=current_iter,
				basis_type='grid_points',
				grid_points=grid_points,
				mc_batch_size=mc_batch_size,
				mc_tol_param=mc_tol,
				normalizing_const_only=False,
				print_error=False)
			
			grad_logpar = mc_output2.reshape(-1, 1)
			
			# compute the gradient of the loss function
			current_grad = grad_logpar - grad_f + self.penalty_param * np.matmul(gram_grid, current_iter)
			
			# compute the updated iter
			new_iter = current_iter - step_size * current_grad
			
			# compute the error of the first update
			grad_new_norm = np.linalg.norm(current_grad, 2)
			error = grad_new_norm / (grad0_norm + 1e-8)
			# np.linalg.norm(new_iter - current_iter, 2) / (np.linalg.norm(current_iter, 2) + 1e-1)
			
			iter_num += 1
			
			if print_error:
				print("Iter = {iter_num}, GradNorm = {gradnorm}, Relative Error = {error}".format(
					iter_num=iter_num, gradnorm=grad_new_norm, error=error))
		
		coefficients = new_iter
		
		return coefficients
	
	def eval_loss_function(self, new_data, coef, basis_type, batchmc_params, grid_points=None, include_penalty=True):
		
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		if len(grid_points.shape) == 1:
			grid_points = grid_points.reshape(-1, 1)
		
		if basis_type not in ['gubasis', 'grid_points']:
			raise NotImplementedError(f"basis_type must be one of 'gubasis' and 'grid_points', but got {basis_type}.")
		
		if basis_type == 'grid_points' and grid_points is None:
			raise ValueError('Please supply grid_points to use grid_points.')
		
		assert self.data.shape[1] == new_data.shape[1], 'The dimensionality of data does not match that of new_data.'
		
		assert self.data.shape[1] == grid_points.shape[
			1], 'The dimensionality of data does not match that of grid_points.'
		
		coef = coef.reshape(-1, 1)
		
		if basis_type == 'gubasis' and coef.shape[0] != self.N:
			
			raise ValueError(("The supplied coef is not correct. "
							  "The length of coef is expected to be {exp_len}, but got {act_len}.").format(
				exp_len=self.N, act_len=len(coef)))
		
		elif basis_type == 'grid_points' and coef.shape[0] != grid_points.shape[0]:
			
			raise ValueError(("The supplied coef is not correct. "
							  "The length of coef is expected to be {exp_len}, but got {act_len}.").format(
				exp_len=grid_points.shape[0], act_len=len(coef)))
		
		mc_batch_size = batchmc_params["mc_batch_size"]
		mc_tol = batchmc_params["mc_tol"]
		
		# compute A(f)
		if basis_type == 'gubasis':
			
			mc_output1 = self.grad_logpar_batchmc(
				coef=coef,
				basis_type='gubasis',
				grid_points=None,
				mc_batch_size=mc_batch_size,
				mc_tol_param=mc_tol,
				normalizing_const_only=True,
				print_error=False)
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function = GaussianPoly2(
					data=np.vstack((self.data, self.contam_data)),
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function = RationalQuadPoly2(
					data=np.vstack((self.data, self.contam_data)),
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			# compute (1 / n) sum_{j=1}^n f (Y_j), where Y_j is the j-th row of new_data
			kernel_mat_new = kernel_function.kernel_gram_matrix(new_data)
			avg_fx = np.mean(np.matmul(kernel_mat_new.T, coef))
			
			# compute the penalty term
			if include_penalty:
				
				gram_mat = kernel_function.kernel_gram_matrix(np.vstack((self.data, self.contam_data)))
				pen_term = self.penalty_param * np.matmul(coef.T, np.matmul(gram_mat, coef)).item() / 2.
			
			else:
				
				pen_term = 0.
		
		elif basis_type == 'grid_points':
			
			mc_output1 = self.grad_logpar_batchmc(
				coef=coef,
				basis_type='grid_points',
				grid_points=grid_points,
				mc_batch_size=mc_batch_size,
				mc_tol_param=mc_tol,
				normalizing_const_only=True,
				print_error=False)
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function_grid = GaussianPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function_grid = RationalQuadPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			# compute (1 / n) sum_{j=1}^n f (Y_j), where Y_j is the j-th row of new_data
			kernel_mat_new = kernel_function_grid.kernel_gram_matrix(new_data)
			avg_fx = np.mean(np.matmul(kernel_mat_new.T, coef))
			
			# compute the penalty term
			if include_penalty:
				
				gram_mat = kernel_function_grid.kernel_gram_matrix(grid_points)
				pen_term = self.penalty_param * np.matmul(coef.T, np.matmul(gram_mat, coef)).item() / 2.
			
			else:
				
				pen_term = 0.
		
		norm_const = mc_output1
		Af = np.log(norm_const)
		
		loss_val = Af - avg_fx + pen_term
		
		return loss_val
	
	def coef_sieve(self, optalgo_params, batchmc_params, start_grid_points=None, step_size_discount_factor=0.5,
				   max_set_grid_points=10, rel_tol_param=1e-3, print_error=True):
		
		"""
		Assume the starting point is the zero vector of the same length as the grid_points.

		"""
		
		if self.d != 1:
			raise ValueError('The function self.coef_sieve only works for the 1-dimensional data.')
		
		if isinstance(step_size_discount_factor, float):
			step_size_discount_factor = [step_size_discount_factor] * (max_set_grid_points - 1)
		
		if isinstance(step_size_discount_factor, list) and len(step_size_discount_factor) != (max_set_grid_points - 1):
			raise ValueError('The length of the step_size_discount_factor should be 1 less than max_set_grid_points.')
		
		# parameters associated with gradient descent algorithm
		step_size = optalgo_params["step_size"]
		max_iter = optalgo_params["max_iter"]
		rel_tol = optalgo_params["rel_tol"]
		
		# first set
		print('=' * 50)
		print('Set 1 of grid points.')
		
		optalgo_params = negloglik_optalgoparams(
			start_pt=np.zeros((start_grid_points.shape[0], 1)),
			step_size=step_size,
			max_iter=max_iter,
			rel_tol=rel_tol)
		
		coef1 = self.coef_grid_points(
			optalgo_params=optalgo_params,
			batchmc_params=batchmc_params,
			grid_points=start_grid_points,
			print_error=print_error)
		
		# evaluate the loss function
		loss1 = self.eval_loss_function(
			new_data=self.data,
			coef=coef1,
			basis_type='grid_points',
			batchmc_params=batchmc_params,
			grid_points=start_grid_points,
			include_penalty=True)
		
		print(f'Negative log-likelihood loss function value = {loss1}.')
		
		# second set
		print('=' * 50)
		print('Set 2 of grid points.')
		
		print(f'step_size_discount_factor={step_size_discount_factor[0]}')
		new_grid_points = finer_grid_points(start_grid_points)
		optalgo_params = negloglik_optalgoparams(
			start_pt=np.zeros((new_grid_points.shape[0], 1)),
			step_size=step_size * step_size_discount_factor[0],
			max_iter=max_iter,
			rel_tol=rel_tol)
		
		coef2 = self.coef_grid_points(
			optalgo_params=optalgo_params,
			batchmc_params=batchmc_params,
			grid_points=new_grid_points,
			print_error=print_error)
		
		# evaluate the loss function
		loss2 = self.eval_loss_function(
			new_data=self.data,
			coef=coef2,
			basis_type='grid_points',
			batchmc_params=batchmc_params,
			grid_points=new_grid_points,
			include_penalty=True)
		
		print(f'Negative log-likelihood loss function value = {loss2}.')
		
		error = np.abs(loss1 - loss2) / (np.abs(loss1) + 1e-8)
		print(f'Relative improvement from increasing the number of basis functions: {error}.')
		j = 2
		
		while error > rel_tol_param and j < max_set_grid_points:
			loss1 = loss2
			
			print('=' * 50)
			print(f'Set {j + 1} of grid points.')
			print(f'step_size_discount_factor={step_size_discount_factor[j - 1]}')
			
			new_grid_points = finer_grid_points(new_grid_points)
			optalgo_params = negloglik_optalgoparams(
				start_pt=np.zeros((new_grid_points.shape[0], 1)),
				step_size=step_size * step_size_discount_factor[j - 1],
				max_iter=max_iter,
				rel_tol=rel_tol)
			
			coef2 = self.coef_grid_points(
				optalgo_params=optalgo_params,
				batchmc_params=batchmc_params,
				grid_points=new_grid_points,
				print_error=print_error)
			
			# evaluate the loss function
			loss2 = self.eval_loss_function(
				new_data=self.data,
				coef=coef2,
				basis_type='grid_points',
				batchmc_params=batchmc_params,
				grid_points=new_grid_points,
				include_penalty=True)
			
			print(f'Negative log-likelihood loss function value = {loss2}.')
			
			error = np.abs(loss1 - loss2) / (np.abs(loss1) + 1e-8)
			print(f'Relative improvement from increasing the number of basis functions: {error}.')
			
			j += 1
		
		if j == max_set_grid_points:
			print('The maximum of the sets of grid points has been reached.')
		
		return coef2, new_grid_points.reshape(-1, self.d)
	
	def natural_param(self, new_data, coef):
		
		# check the validity of new_data
		if isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		# check compatibility of data and contam_data
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		new_data_N, new_data_d = new_data.shape
		
		if new_data_d != self.d:
			raise ValueError('The shape of new_data is not compatible with self.data and self.contam_data.')
		
		if not isinstance(coef[-1], str):
			raise ValueError(('coef should be the direct output of self.gubasis, self.grid_points, or self.sieve, '
							  'and its last item should be a str. Please double check.'))
		
		basis_type = coef[-1]
		
		if basis_type == 'gubasis':
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function = GaussianPoly2(
					data=np.vstack((self.data, self.contam_data)),
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function = RationalQuadPoly2(
					data=np.vstack((self.data, self.contam_data)),
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			f_matrix = kernel_function.kernel_gram_matrix(new_data)
			coef_vec = coef[0]
		
		elif basis_type == 'grid_points' or basis_type == 'sieve':
			
			coef_vec = coef[0]
			grid_points = coef[1]
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function_grid = GaussianPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function_grid = RationalQuadPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			f_matrix = kernel_function_grid.kernel_gram_matrix(new_data)
		
		output = np.matmul(f_matrix.T, coef_vec).flatten()
		
		return output
	
	def unnormalized_density_eval_1d(self, x, coef):
		
		if self.d != 1:
			error_msg = (f'The function self.unnormalized_density_eval_1d only works for 1-dimensional data. '
						 f'But the underlying data is {self.d}-dimensional.')
			raise ValueError(error_msg)
		
		if not isinstance(coef[-1], str):
			raise ValueError(('coef should be the direct output of self.gubasis, self.grid_points, or self.sieve, '
							  'and its last item should be a str. Please double check.'))
		
		basis_type = coef[-1]
		
		if basis_type == 'gubasis':
			
			n_basis = self.data.shape[0]
			coef_vec = coef[0]
			
			den = (self.base_density.baseden_eval_1d(x) *
				   np.exp(np.sum([coef_vec[i] * self.kernel_function_data.kernel_x_1d(self.data[i,])(x)
								  for i in range(n_basis)]) +
						  coef_vec[-1] * self.kernel_function_contam_data.kernel_x_1d(self.contam_data)(x)))
		
		elif basis_type == 'grid_points' or basis_type == 'sieve':
			
			coef_vec = coef[0]
			grid_points = coef[1]
			n_basis = len(grid_points)
			
			if self.kernel_type == 'gaussian_poly2':
				
				kernel_function_grid = GaussianPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			elif self.kernel_type == 'rationalquad_poly2':
				
				kernel_function_grid = RationalQuadPoly2(
					data=grid_points,
					r1=self.r1,
					r2=self.r2,
					c=self.c,
					bw=self.bw)
			
			den = (self.base_density.baseden_eval_1d(x) *
				   np.exp(np.sum([coef_vec[i] * kernel_function_grid.kernel_x_1d(grid_points[i,])(x)
								  for i in range(n_basis)])))
		
		return den
	
	def density_logpartition_1d(self, coef):
		
		if self.d != 1:
			error_msg = (f'The function self.density_logpartition_1d only works for 1-dimensional data. '
						 f'But the underlying data is {self.d}-dimensional.')
			raise ValueError(error_msg)
		
		norm_const, _ = integrate.quad(
			func=self.unnormalized_density_eval_1d,
			a=self.base_density.domain[0][0],
			b=self.base_density.domain[0][1],
			args=(coef,),
			limit=100)
		
		output = np.log(norm_const)
		
		return output
	
	def log_density(self, new_data, coef, compute_base_density=False):
		
		if compute_base_density:
			
			baseden_part = np.log(self.base_density.baseden_eval(new_data).flatten())
		
		else:
			
			baseden_part = 0.
		
		natparam = self.natural_param(
			new_data=new_data,
			coef=coef)
		logpar = self.density_logpartition_1d(coef=coef)
		
		output = baseden_part + natparam - logpar
		
		return output
