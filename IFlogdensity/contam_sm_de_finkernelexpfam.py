import numpy as np
from dekef.kernel_function import *
from dekef.base_density import *
from dekef.check import *
from scipy import integrate


class ContamSMFinKernelExpFam:
	
	"""
	A class of estimating the probability density function using the score matching loss function
	in the presence of a contaminated observation.
	We assume the basis functions in the natural parameter are the kernel functions centered at self.grid_points.
	
	...
	
	Attributes
	----------
	data : numpy.ndarray
		The array of uncontaminated observations.
	
	contam_data : numpy.ndarray
		The contaminated observation.
	
	grid_points : numpy.ndarray
		The array of grid points at which the basis functions of the natural parameter are centered.
	
	N : int
		The number of uncontaminated observations.
		
	n : int
		The number of contaminated observation; should always be 1.
	
	d : int
		The dimensionality of self.data and self.contam_data.
	
	n_basis : int
		The number of basis functions, i.e., the length of self.grid_points.
	
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
	
	kernel_function_data : kernel_function
		The class of kernel functions centered at self.data.
	
	kernel_function_contam_data : kernel_function
		The class of kernel functions centered at self.contam_data.
	
	kernel_function_grid_points : kernel_function
		The class of kernel functions centered at self.grid_points.
	
	gram_matrix : numpy.ndarray
		The Gram matrix of shape (self.n_basis, self.n_basis) whose (i, j)-entry is the inner product
		between k (w_i, .) and k (w_j, .), where w_i and w_j are the i-th and j-th rows of self.grod_points,
		respectively.
	
	Methods
	-------
	matrix_G()
		Returns a matrix of shape self.n_basis by (self.N * self.d) with (j, (i-1)d+u)-entry being
		the inner product of k (w_j, \cdot) and \partial_u k (X_i, \cdot),
		where w_j is the j-th row of self.grod_points, and X_i is the i-th row of self.data.
		
	matrix_H()
		Returns a matrix of shape self.n_basis by self.d with (j, u)-entry being
		the inner product of k (w_j, \cdot) and \partial_u k (y, \cdot),
		where w_j is the j-th row of self.grod_points, and y is self.contam_data.
	
	vector_t1()
		Returns a vector of shape self.n_basis by 1 with j-th entry being
			- \frac{1}{n} \sum_{i=1}^n \sum_{u=1}^d
				   (\partial_u^2 k (X_i, w_j) + (\partial_u \log \mu) (X_i) \partial_u k (X_i, w_j))
		where w_j is the j-th row of self.grid_points, X_i is the i-th row of self.data,
		and mu is the base density function.
		
	vector_t2()
		Returns a vector of shape self.n_basis by 1 with j-th entry being
		- \sum_{u=1}^d (\partial_u^2 k (y, w_j) + (\partial_u \log \mu) (y) \partial_u k (y, w_j)),
		where w_j is the j-th row of self.grid_points, y is self.contam_data, and mu is the base density function.
		
	coef()
		Computes the coefficient vector of basis functions in the natural parameter
		that minimizes the penalized score matching loss function.
	
	natural_param(new_data, coef)
		Evaluates the natural parameter in the score matching density estimate at new_data.
	
	unnormalized_density_eval_1d(x, coef)
		Evaluates the density function up to a normalizing constant at 1-dimensional data x.
		
	density_logpartition_1d(coef)
		Evaluates the log-partition function at coef.
	
	log_density(new_data, coef, compute_base_density=False)
		Evaluates the log-density function at new_data.
		
	"""
	
	def __init__(self, data, contam_data, grid_points, contam_weight, penalty_param, base_density,
				 r1=1.0, r2=0.0, c=0., bw=1., kernel_type='gaussian_poly2'):
		
		"""
		Parameters
		----------
		data : numpy.ndarray
			The array of observations whose probability density function is to be estimated.

		contam_data : numpy.ndarray
			The array of contaminated observation.
			
		grid_points : numpy.ndarray
			The array of grid points at which the basis functions of the natural parameter are centered.

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
		
		# check types of data and contam_data
		if not isinstance(data, np.ndarray):
			data = np.array(data)
		
		if not isinstance(contam_data, np.ndarray):
			contam_data = np.array(contam_data)
		
		if not isinstance(grid_points, np.ndarray):
			grid_points = np.array(grid_points)
		
		# check compatibility of data and contam_data
		if len(data.shape) == 1:
			data = data.reshape(-1, 1)
		
		if len(contam_data.shape) == 1:
			contam_data = contam_data.reshape(-1, 1)
		
		if len(grid_points.shape) == 1:
			grid_points = grid_points.reshape(-1, 1)
		
		N, d = data.shape
		n, d1 = contam_data.shape
		m, d2 = grid_points.shape
		
		if d != d1 or d != d2 or d1 != d2:
			raise ValueError('The shapes of data, contam_data and grid_points are not compatible.')
		
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
		self.grid_points = grid_points
		self.N = N
		self.n = n
		self.d = d
		self.n_basis = m
		
		# check the validity of the contam_weight
		assert 0. <= contam_weight <= 1., "contam_weight must be between 0 and 1, inclusively."
		self.contam_weight = contam_weight
		
		# check the validity of the penalty_param
		assert penalty_param > 0., "penalty_param must be strictly positive."
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
			
			self.kernel_function_grid_points = GaussianPoly2(
				data=grid_points,
				r1=self.r1,
				r2=self.r2,
				c=self.c,
				bw=self.bw)
			
			self.gram_matrix = self.kernel_function_grid_points.kernel_gram_matrix(
				new_data=self.grid_points)
		
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
			
			self.gram_matrix = self.kernel_function_grid_points.kernel_gram_matrix(
				new_data=self.grid_points)
	
	def matrix_G(self):
		
		"""
		Returns a matrix of shape self.n_basis by (self.N * self.d) with (j, (i-1)d+u)-entry being
		the inner product of k (w_j, \cdot) and \partial_u k (X_i, \cdot),
		where w_j is the j-th row of self.grod_points, and X_i is the i-th row of self.data.
		
		Returns
		-------
		numpy.ndarray
			An array of shape self.n_basis by (self.N * self.d). Please see details above.

		"""
		
		output = self.kernel_function_data.partial_kernel_matrix_10(new_data=self.grid_points).T
		
		return output
	
	def matrix_H(self):
		
		"""
		Returns a matrix of shape self.n_basis by self.d with (j, u)-entry being
		the inner product of k (w_j, \cdot) and \partial_u k (y, \cdot),
		where w_j is the j-th row of self.grod_points, and y is self.contam_data.
		
		Returns
		-------
		numpy.ndarray
			An array of shape self.n_basis by self.d. Please see details above.
			
		"""
		
		output = self.kernel_function_contam_data.partial_kernel_matrix_10(new_data=self.grid_points).T
		
		return output
	
	def vector_t1(self):
		
		"""
		Returns a vector of shape self.n_basis by 1 with j-th entry being
			- \frac{1}{n} \sum_{i=1}^n \sum_{u=1}^d
				   (\partial_u^2 k (X_i, w_j) + (\partial_u \log \mu) (X_i) \partial_u k (X_i, w_j))
		where w_j is the j-th row of self.grid_points, X_i is the i-th row of self.data,
		and mu is the base density function.
		
		Returns
		-------
		numpy.ndarray
			An array of shape self.n_basis by 1. Please see details above.
			
		"""
		
		kernel_partial_10 = self.kernel_function_data.partial_kernel_matrix_10(new_data=self.grid_points)
		kernel_partial_20 = self.kernel_function_data.partial_kernel_matrix_20(new_data=self.grid_points)
		
		# partial derivatives of log base density
		baseden_partial = np.zeros(self.data.shape, dtype=np.float64)
		for u in range(self.data.shape[1]):
			baseden_partial[:, u] = self.base_density.logbaseden_deriv1(new_data=self.data, j=u).flatten()
		
		baseden_partial = baseden_partial.flatten()
		
		hatz1 = -np.sum(kernel_partial_10 * baseden_partial[:, np.newaxis], axis=0).reshape(1, -1) / self.data.shape[0]
		hatz2 = -np.sum(kernel_partial_20, axis=0).reshape(1, -1) / self.data.shape[0]
		hatz = hatz1 + hatz2
		
		return hatz.T
	
	def vector_t2(self):
		
		"""
		Returns a vector of shape self.n_basis by 1 with j-th entry being
		- \sum_{u=1}^d (\partial_u^2 k (y, w_j) + (\partial_u \log \mu) (y) \partial_u k (y, w_j)),
		where w_j is the j-th row of self.grid_points, y is self.contam_data, and mu is the base density function.
		
		Returns
		-------
		numpy.ndarray
			An array of shape self.n_basis by 1. Please see details above.
		
		"""
		
		kernel_partial_10 = self.kernel_function_contam_data.partial_kernel_matrix_10(new_data=self.grid_points)
		kernel_partial_20 = self.kernel_function_contam_data.partial_kernel_matrix_20(new_data=self.grid_points)
		
		# partial derivatives of log base density
		baseden_partial = np.zeros(self.contam_data.shape, dtype=np.float64)
		for u in range(self.contam_data.shape[1]):
			baseden_partial[:, u] = self.base_density.logbaseden_deriv1(new_data=self.contam_data, j=u).flatten()
		
		baseden_partial = baseden_partial.flatten()
		
		hatz1 = -np.sum(kernel_partial_10 * baseden_partial[:, np.newaxis], axis=0).reshape(1, -1)
		hatz2 = -np.sum(kernel_partial_20, axis=0).reshape(1, -1)
		hatz = hatz1 + hatz2
		
		return hatz.T
	
	def coef(self):
		
		"""
		Computes the coefficient vector of basis functions in the natural parameter
		that minimizes the penalized score matching loss function.
		
		Returns
		-------
		numpy.ndarray
			An array of coefficient vector of shape self.n_basis by 1.
			
		"""
		
		matG = self.matrix_G()
		v1 = self.vector_t1()
		matK = self.gram_matrix
		
		matH = self.matrix_H()
		v2 = self.vector_t2()
		
		lhs = ((1. - self.contam_weight) * matG @ matG.T / self.N +
			   self.contam_weight * matH @ matH.T +
			   self.penalty_param * matK)
		rhs = (1. - self.contam_weight) * v1 + self.contam_weight * v2
		output = np.linalg.solve(lhs, rhs)
		
		return output
	
	def natural_param(self, new_data, coef):
		
		"""
		Evaluates the natural parameter in the score matching density estimate at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			The array of data at which the natural parameter is to be evaluated.
			
		coef : numpy.ndarray
			The coefficient vector of basis functions in the natural parameter.
		
		Returns
		-------
		numpy.ndarray
			The 1-dimensional array of the values of the natural parameter estimates at new_data.
		
		"""
		
		# check the validity of new_data
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		# check compatibility of data and contam_data
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		new_data_N, new_data_d = new_data.shape
		
		if new_data_d != self.d:
			raise ValueError('The shape of new_data is not compatible with self.data and self.contam_data.')
		
		f_matrix = self.kernel_function_grid_points.kernel_gram_matrix(new_data)
		output = np.matmul(f_matrix.T, coef).flatten()
		
		return output
	
	def unnormalized_density_eval_1d(self, x, coef):
		
		"""
		Evaluates the density function up to a normalizing constant at 1-dimensional data x.
		This function is mainly used in computing the normalizing constant and only works when self.d is equal to 1.
		
		Parameters
		----------
		x : float or numpy.ndarray
			The point at which the un-normalized density function is to be evaluated.
			
		coef : numpy.ndarray
			The coefficient vector of basis functions in the natural parameter.
		
		Returns
		-------
		float or numpy.ndarray
			The value of the un-normalized density function at x.
		
		"""
		
		if self.d != 1:
			error_msg = (f'The function self.unnormalized_density_eval_1d only works for 1-dimensional data. '
						 f'But the underlying data is {self.d}-dimensional.')
			raise ValueError(error_msg)
		
		den = (self.base_density.baseden_eval_1d(x) * np.exp(
			np.sum([coef[i] * self.kernel_function_grid_points.kernel_x_1d(self.grid_points[i, ])(x)
					for i in range(self.n_basis)])))
		
		return den
	
	def density_logpartition_1d(self, coef):
		
		"""
		Evaluates the log-partition function at coef.
		
		Parameters
		----------
		coef : numpy.ndarray
			The coefficient vector of basis functions in the natural parameter.
		
		Returns
		-------
		float
			The value of the log-partition function at coef.
			
		"""
		
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
		
		"""
		Evaluates the log-density function at new_data.
		
		Parameters
		----------
		new_data : numpy.ndarray
			The array of data at which the log-density function is to be evaluated.
			
		coef : numpy.ndarray
			The coefficient vector of basis functions in the natural parameter.
		
		compute_base_density : bool, optional
			Whether to compute the base density part; default is False.
		
		Returns
		-------
		numpy.ndarray
			An 1-dimensional array of the values of the log-density function at new_data.
		 
		"""
		
		# check the validity of new_data
		if not isinstance(new_data, np.ndarray):
			new_data = np.array(new_data)
		
		# check compatibility of data and contam_data
		if len(new_data.shape) == 1:
			new_data = new_data.reshape(-1, 1)
		
		new_data_N, new_data_d = new_data.shape
		
		if new_data_d != self.d:
			raise ValueError('The shape of new_data is not compatible with self.data and self.contam_data.')
		
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
