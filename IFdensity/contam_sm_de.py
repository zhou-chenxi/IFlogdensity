import numpy as np
from dekef.base_density import *
from dekef.kernel_function import *
from dekef.scorematching_common_functions import *
from scipy import integrate


class ContamSMDensityEstimate:
	
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
		assert penalty_param > 0., "penalty_param must be between 0 and 1, inclusively."
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
	
	def matrix_K11(self):
		
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
		
		pass
	
	# return K33
	
	def matrix_K34(self):
		
		pass
	
	# return K34
	
	def matrix_K43(self):
		
		pass
	
	# return K43
	
	def matrix_K44(self):
		
		pass
	
	# return K44
	
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