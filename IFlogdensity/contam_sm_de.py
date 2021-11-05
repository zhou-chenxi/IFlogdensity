import numpy as np
from dekef.base_density import *
from dekef.kernel_function import *
from dekef.scorematching_common_functions import *
from scipy import integrate


class ContamSMDensityEstimate:
	
	"""
	A class of estimating the probability density function using the score matching loss function
	in the presence of a contaminated observation.
	
	...
	
	Attributes
	----------
	data : numpy.ndarray
		The array of uncontaminated observations whose density function is to be estimated.
	
	contam_data : numpy.ndarray
		The contaminated observation.
	
	N : int
		The number of uncontaminated observations.
		
	n : int
		The number of contaminated observation; should always be 1.
	
	d : int
		The dimensionality of self.data and self.contam_data.
	
	contam_weight : float
		The weight of self.contam_data; must always be between 0 and 1, inclusively.
	
	penalty_param : float
		The penalty parameter used to compute the penalized score matching density estimate;
		must be strictly positive.
		
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
		The type of the kernel function used; must be either 'gaussian_poly2' or 'rationalquad_poly2'.
	
	kernel_function_data : kernel_function
	
	
	
	kernel_function_contam_data : kernel_function
	
	
	Methods
	-------
	matrix_K11()
		Evaluates a matrix of size (self.N * self.d) by (self.N * self.d)
		with the ((i-1)d+u, (j-1)d+v)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and \partial_v k(X_j, \cdot),
		where X_i is the i-th observation in self.data.
	
	matrix_K12()
		Evaluates a matrix of size (self.N * self.d) by self.d
		with the ((i-1)d+u, v)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and \partial_v k(y, \cdot),
		where X_i is the i-th observation in self.data and y is self.contam_data.
	
	matrix_K21()
		Evaluates a matrix of size self.d by (self.N * self.d)
		with the (v, (i-1)d+u)-th entry being the inner product
		between \partial_v k(y, \cdot) and \partial_u k(X_i, \cdot),
		where X_i is the i-th observation in self.data and y is self.contam_data.
	
	matrix_K13()
		Evaluates a matrix of size (self.N * self.d) by 1
		with the ((i-1)d+u)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and z_{F_n},
		where
		z_{F_n} = \frac{1}{N} \sum_{i=1}^N \sum_{u=1}^d (\partial_u^2 k(X_i, \cdot) +
		             (\partial_u \log \mu) (X_i) \partial_u k (X_i, \cdot)),
		X_i is the i-th observation in self.data, \mu is the base density, N = self.N, d = self.d.
		
	matrix_K31()
		Evaluates a matrix of size 1 by (self.N * self.d)
		with the ((i-1)d+u)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and z_{F_n},
		where
		z_{F_n} = \frac{1}{N} \sum_{i=1}^N \sum_{u=1}^d (\partial_u^2 k(X_i, \cdot) +
		             (\partial_u \log \mu) (X_i) \partial_u k (X_i, \cdot)),
		X_i is the i-th observation in self.data, \mu is the base density, N = self.N, d = self.d.
	
	matrix_K14()
		Evaluates a matrix of size (self.N * self.d) by 1
		with the ((i-1)d+u)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and z_{\delta_y},
		where
		z_{\delta_y} = \sum_{u=1}^d (\partial_u^2 k(y, \cdot) +
		             (\partial_u \log \mu) (y) \partial_u k (y, \cdot)),
		X_i is the i-th observation in self.data, y is self.contam_data,
		\mu is the base density, d = self.d.
	
	matrix_K41()
		Evaluates a matrix of size 1 by (self.N * self.d)
		with the ((i-1)d+u)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and z_{\delta_y},
		where
		z_{\delta_y} = \sum_{u=1}^d (\partial_u^2 k(y, \cdot) +
		             (\partial_u \log \mu) (y) \partial_u k (y, \cdot)),
		X_i is the i-th observation in self.data, y is self.contam_data,
		\mu is the base density, d = self.d.
		
	matrix_K22()
		Evaluates a matrix of size self.d by self.d
		with the (u, v)-th entry being the inner product
		between \partial_u k(y, \cdot) and \partial_v k(y, \cdot),
		where y is self.contam_data.
		
	matrix_K23()
		Evaluates a matrix of size self.d by self.d
		with the (u, v)-th entry being the inner product
		between \partial_u k(y, \cdot) and \partial_v k(y, \cdot),
		where
		
		
		
		y is self.contam_data.
		
	
	matrix_K32()
	
	matrix_K24()
	
	matrix_K42()
	
	matrix_K33()
	
	matrix_K34()
	
	
	matrix_K43()
	
	matrix_K44()
	
	
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
	
	def matrix_K11(self):
		
		"""
		Evaluates the matrix K11, a matrix of size (self.N * self.d) by (self.N * self.d)
		with the ((i-1)d+u, (j-1)d+v)-th entry being the inner product
		between \partial_u k(X_i, \cdot) and \partial_v k(X_j, \cdot).z

		Returns
		-------
		
		"""
		
		
		K11 = self.kernel_function_data.partial_kernel_matrix_11(new_data=self.data)
		
		return K11
	
	def matrix_K12(self):
		
		K12 = self.kernel_function_data.partial_kernel_matrix_11(new_data=self.contam_data)
		
		return K12
	
	def matrix_K21(self):
		
		K21 = self.matrix_K12().T
		
		return K21
	
	def matrix_K13(self):
		
		K13 = vector_h(
			data=self.data,
			kernel_function=self.kernel_function_data,
			base_density=self.base_density)
		
		return K13
	
	def matrix_K31(self):
		
		K31 = self.matrix_K13().T
		
		return K31
	
	def matrix_K14(self):
		
		# \partial_u \partial_v k(X_i, y), result is nd * d
		kernel_partial_11 = self.kernel_function_data.partial_kernel_matrix_11(new_data=self.contam_data)
		
		# <\hat{z}_{\delta_y}, \partial_u k(X_i, \cdot)>, result is nd * d
		kernel_partial_12 = self.kernel_function_data.partial_kernel_matrix_12(new_data=self.contam_data)
		baseden_partial = np.zeros(self.contam_data.shape, dtype=np.float64)
		
		for u in range(self.contam_data.shape[1]):
			baseden_partial[:, u] = self.base_density.logbaseden_deriv1(new_data=self.contam_data, j=u).flatten()
		
		baseden_partial = baseden_partial.reshape(-1, 1)
		
		h1 = np.sum(kernel_partial_12, axis=1).reshape(-1, 1) / self.contam_data.shape[0]
		h2 = np.sum(np.matmul(kernel_partial_11, baseden_partial), axis=1).reshape(-1, 1) / self.contam_data.shape[0]
		K14 = -(h1 + h2)  # the negative sign comes from the \hat{z} part
		return K14
	
	def matrix_K41(self):
		
		K41 = self.matrix_K14().T
		
		return K41
	
	def matrix_K22(self):
		
		K22 = self.kernel_function_contam_data.partial_kernel_matrix_11(new_data=self.contam_data)
		
		return K22
	
	def matrix_K23(self):
		
		# \partial_u \partial_v k(y, X_j), result is d * nd
		kernel_partial_11 = self.kernel_function_contam_data.partial_kernel_matrix_11(new_data=self.data)
		
		# <\hat{z}_{F_n}, \partial_u k(y, \cdot)>, result is d * nd
		kernel_partial_12 = self.kernel_function_contam_data.partial_kernel_matrix_12(new_data=self.data)
		
		baseden_partial = np.zeros(self.data.shape, dtype=np.float64)
		
		for u in range(self.data.shape[1]):
			baseden_partial[:, u] = self.base_density.logbaseden_deriv1(new_data=self.data, j=u).flatten()
		
		baseden_partial = baseden_partial.reshape(-1, 1)
		
		h1 = np.sum(kernel_partial_12, axis=1).reshape(-1, 1) / self.data.shape[0]
		h2 = np.sum(np.matmul(kernel_partial_11, baseden_partial), axis=1).reshape(-1, 1) / self.data.shape[0]
		
		K23 = -(h1 + h2)  # the negative sign comes from the \hat{z} part
		return K23
	
	def matrix_K32(self):
		
		K32 = self.matrix_K23().T
		
		return K32
	
	def matrix_K24(self):
		
		K24 = vector_h(
			data=self.contam_data,
			kernel_function=self.kernel_function_contam_data,
			base_density=self.base_density)
		
		return K24
	
	def matrix_K42(self):
		
		K42 = self.matrix_K24().T
		
		return K42
	
	def matrix_K33(self):
		
		kernel_partial_11 = self.kernel_function_data.partial_kernel_matrix_11(new_data=self.data)
		
		baseden_partial = np.zeros(self.data.shape, dtype=np.float64)
		for u in range(self.data.shape[1]):
			baseden_partial[:, u] = self.base_density.logbaseden_deriv1(new_data=self.data, j=u).flatten()
		baseden_partial = baseden_partial.reshape(-1, 1)
		
		z_norm1 = np.matmul(baseden_partial.T, np.matmul(kernel_partial_11, baseden_partial))
		
		kernel_partial_12 = self.kernel_function_data.partial_kernel_matrix_12(new_data=self.data)
		z_norm2 = np.sum(baseden_partial * kernel_partial_12)
		
		kernel_partial_21 = self.kernel_function_data.partial_kernel_matrix_21(new_data=self.data)
		z_norm3 = np.sum(np.matmul(kernel_partial_21, baseden_partial))
		
		z_norm4 = np.sum(self.kernel_function_data.partial_kernel_matrix_22(new_data=self.data))
		K33 = (z_norm1 + z_norm2 + z_norm3 + z_norm4) / self.N ** 2
		
		return K33
	
	def matrix_K34(self):
		
		kernel_partial_11 = self.kernel_function_data.partial_kernel_matrix_11(new_data=self.contam_data)
		
		baseden_partial_data = np.zeros(self.data.shape, dtype=np.float64)
		for u in range(self.data.shape[1]):
			baseden_partial_data[:, u] = self.base_density.logbaseden_deriv1(
				new_data=self.data, j=u).flatten()
		baseden_partial_data = baseden_partial_data.reshape(-1, 1)
		
		baseden_partial_contam_data = np.zeros(self.contam_data.shape, dtype=np.float64)
		for u in range(self.contam_data.shape[1]):
			baseden_partial_contam_data[:, u] = self.base_density.logbaseden_deriv1(
				new_data=self.contam_data, j=u).flatten()
		baseden_partial_contam_data = baseden_partial_contam_data.reshape(-1, 1)
		
		z_norm1 = np.matmul(baseden_partial_data.T, np.matmul(kernel_partial_11, baseden_partial_contam_data))
		
		kernel_partial_12 = self.kernel_function_data.partial_kernel_matrix_12(new_data=self.contam_data)
		z_norm2 = np.sum(baseden_partial_data * kernel_partial_12)
		
		kernel_partial_21 = self.kernel_function_data.partial_kernel_matrix_21(new_data=self.contam_data)
		# print(kernel_partial_21.shape, baseden_partial_contam_data.shape)
		z_norm3 = np.sum(np.matmul(kernel_partial_21, baseden_partial_contam_data))
		
		z_norm4 = np.sum(self.kernel_function_data.partial_kernel_matrix_22(new_data=self.contam_data))
		K34 = (z_norm1 + z_norm2 + z_norm3 + z_norm4) / self.N / self.n
		
		return K34
	
	def matrix_K43(self):
		
		K43 = self.matrix_K34()
		
		return K43
	
	def matrix_K44(self):
		
		kernel_partial_11 = self.kernel_function_contam_data.partial_kernel_matrix_11(new_data=self.contam_data)
		
		kernel_partial_12 = self.kernel_function_contam_data.partial_kernel_matrix_12(new_data=self.contam_data)
		baseden_partial = np.zeros(self.contam_data.shape, dtype=np.float64)
		for u in range(self.contam_data.shape[1]):
			baseden_partial[:, u] = self.base_density.logbaseden_deriv1(new_data=self.contam_data, j=u).flatten()
		baseden_partial = baseden_partial.reshape(-1, 1)
		
		z_norm1 = np.matmul(baseden_partial.T, np.matmul(kernel_partial_11, baseden_partial))
		
		kernel_partial_12 = self.kernel_function_contam_data.partial_kernel_matrix_12(new_data=self.contam_data)
		z_norm2 = np.sum(baseden_partial * kernel_partial_12)
		
		kernel_partial_21 = self.kernel_function_contam_data.partial_kernel_matrix_21(new_data=self.contam_data)
		z_norm3 = np.sum(np.matmul(kernel_partial_21, baseden_partial))
		
		z_norm4 = np.sum(self.kernel_function_contam_data.partial_kernel_matrix_22(new_data=self.contam_data))
		K44 = (z_norm1 + z_norm2 + z_norm3 + z_norm4) / self.n ** 2
		
		return K44
	
	def matrix_K(self):
	
		K11 = self.matrix_K11()
		K12 = self.matrix_K12()
		K13 = self.matrix_K13()
		K14 = self.matrix_K14()
		
		K21 = self.matrix_K21()
		K22 = self.matrix_K22()
		K23 = self.matrix_K23()
		K24 = self.matrix_K24()
		
		K31 = self.matrix_K31()
		K32 = self.matrix_K32()
		K33 = self.matrix_K33()
		K34 = self.matrix_K34()
		
		K41 = self.matrix_K41()
		K42 = self.matrix_K42()
		K43 = self.matrix_K43()
		K44 = self.matrix_K44()
		
		row1 = np.hstack((K11, K12, K13, K14))
		row2 = np.hstack((K21, K22, K23, K24))
		row3 = np.hstack((K31, K32, K33, K34))
		row4 = np.hstack((K41, K42, K43, K44))
		
		output = np.vstack((row1, row2, row3, row4))
		
		return output
	
	def coef(self):
		
		# form the large matrix to compute coef_i for all i = 0, \cdots, nd+d-1
		# LHS
		K11_sol = ((1 - self.contam_weight) * self.matrix_K11() / self.N +
				   self.penalty_param * np.eye(self.N * self.d, dtype=np.float64))
		K12_sol = (1 - self.contam_weight) * self.matrix_K12() / self.N
		K21_sol = self.contam_weight * self.matrix_K21()
		K22_sol = (self.contam_weight * self.matrix_K22() +
				   self.penalty_param * np.eye(self.n * self.d, dtype=np.float64))
		
		K1112_sol = np.hstack((K11_sol, K12_sol))
		K2122_sol = np.hstack((K21_sol, K22_sol))
		
		K_sol = np.vstack((K1112_sol, K2122_sol))
		
		# RHS
		b1 = - ((1 - self.contam_weight) ** 2 * self.matrix_K13() +
				self.contam_weight * (1 - self.contam_weight) * self.matrix_K14()) / (self.N * self.penalty_param)
		
		b2 = - (self.contam_weight * (1 - self.contam_weight) * self.matrix_K23() +
				self.contam_weight ** 2 * self.matrix_K24()) / self.penalty_param
		
		b = np.vstack((b1, b2))
		
		# solve the linear system
		result1 = np.linalg.lstsq(K_sol, b, rcond=None)[0].reshape(-1, 1)
		
		# combine the coefficients
		output = np.vstack((
			result1,
			(1 - self.contam_weight) / self.penalty_param,
			self.contam_weight / self.penalty_param))
		
		return output
	
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
		
		f_matrix_data = kernel_partial10_hatz(
			data=self.data,
			new_data=new_data,
			kernel_function=self.kernel_function_data,
			base_density=self.base_density)
		f_matrix_data1 = f_matrix_data[:(self.N * self.d)]
		f_matrix_data2 = f_matrix_data[[-1]]
		
		f_matrix_contam_data = kernel_partial10_hatz(
			data=self.contam_data,
			new_data=new_data,
			kernel_function=self.kernel_function_contam_data,
			base_density=self.base_density)
		f_matrix_contam_data1 = f_matrix_contam_data[:(self.n * self.d)]
		f_matrix_contam_data2 = f_matrix_contam_data[[-1]]
		
		f_matrix = np.vstack((
			f_matrix_data1,
			f_matrix_contam_data1,
			f_matrix_data2,
			f_matrix_contam_data2))
		
		output = np.matmul(f_matrix.T, coef).flatten()
		
		return output
	
	def unnormalized_density_eval_1d(self, x, coef):
		
		if self.d != 1:
			error_msg = (f'The function self.unnormalized_density_eval_1d only works for 1-dimensional data. '
						 f'But the underlying data is {self.d}-dimensional.')
			raise ValueError(error_msg)
		
		# linear combination of first derivatives at data
		fx1_data = np.sum([coef[i] * self.kernel_function_data.kernel_x_1d_deriv1(self.data[i,])(x)
						   for i in range(self.N)])
		
		# linear combination of first derivatives at contam_data
		fx1_contam_data = (coef[self.N] *
						   self.kernel_function_contam_data.kernel_x_1d_deriv1(self.contam_data)(x))
		
		# z part involving data
		z_data_1 = np.sum([self.base_density.logbaseden_deriv1(new_data=self.data[i,].reshape(1, 1), j=0) *
						   self.kernel_function_data.kernel_x_1d_deriv1(self.data[i,])(x)
						   for i in range(self.N)])
		z_data_2 = np.sum([self.kernel_function_data.kernel_x_1d_deriv2(self.data[i,])(x)
						   for i in range(self.N)])
		z_data = -(z_data_1 + z_data_2) / self.N
		
		# z part involving contam_data
		z_cdata_1 = (self.base_density.logbaseden_deriv1(new_data=self.contam_data.reshape(1, 1), j=0) *
					 self.kernel_function_contam_data.kernel_x_1d_deriv1(self.contam_data)(x))
		z_cdata_2 = self.kernel_function_contam_data.kernel_x_1d_deriv2(self.contam_data)(x)
		z_cdata = -(z_cdata_1 + z_cdata_2)
		
		natparam = fx1_data + fx1_contam_data + coef[self.N + 1] * z_data + coef[self.N + 2] * z_cdata
		output = self.base_density.baseden_eval_1d(x) * np.exp(natparam)
		
		return output
	
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
	
	def log_density(self, new_data, compute_base_density=False):
		
		if compute_base_density:
			baseden_part = np.log(self.base_density.baseden_eval(new_data).flatten())
		else:
			baseden_part = 0.
		
		coef = self.coef()
		natparam = self.natural_param(
			new_data=new_data,
			coef=coef)
		logpar = self.density_logpartition_1d(coef=coef)
		
		output = baseden_part + natparam - logpar
		
		return output
