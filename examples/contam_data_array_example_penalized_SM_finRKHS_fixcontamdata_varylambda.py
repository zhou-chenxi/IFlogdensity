#!/usr/bin/env python3

import os
import copy
import numpy as np
from dekef.base_density import *
from IFlogdensity.influence_function_sm_de import plot_IF_1d_params
from IFlogdensity.contam_sm_de_finkernelexpfam import *


if __name__ == '__main__':
    
    os.chdir('/Users/chenxizhou/Dropbox/code_package/IFlogdensity')
    true_data = np.load('data/geyser.npy').astype(np.float64)
    df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
    
    # original data with 108.0 removed
    data_waiting = df[df != 108.0]
    
    # array of contaminated data
    contam_data = np.array([120.]).reshape(-1, 1)
    
    print(f'Contaminated observation = {contam_data.item()}.')

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 7.0
    
    # penalty parameter
    log_pen_param_list = np.arange(-15., 1.5, 0.5)
    
    print(f'bw={bw}.')
    
    # contamnation weight
    contam_weight = 1e-3
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 310.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']

    new_data = np.linspace(plot_xlimit[0], plot_xlimit[1], plot_pts_cnt)

    full_save_folder = (f'data/PenSM-FinKEF-ContamData={contam_data}-bw={bw}-kernel={kernel_type}-' +
                        f'contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_pts_cnt}')
    if not os.path.isdir(full_save_folder):
        os.mkdir(full_save_folder)
        
    r1, r2, c = 1., 0., 0.
    grid_points = np.arange(1., 311., 1)
    
    for log_pen_param in log_pen_param_list:
        
        print('*' * 50)
        print(f'log penalty = {log_pen_param}')

        uncontam_sm = ContamSMFinKernelExpFam(
            data=data_waiting,
            contam_data=contam_data.reshape(-1, 1),
            grid_points=grid_points,
            contam_weight=0.,
            penalty_param=np.exp(log_pen_param),
            base_density=base_density,
            r1=r1,
            r2=r2,
            c=c,
            bw=bw,
            kernel_type='gaussian_poly2')

        uncontam_coef = uncontam_sm.coef()

        gram_mat = uncontam_sm.gram_matrix
        uncontam_norm = np.sqrt(uncontam_coef.T @ gram_mat @ uncontam_coef).item()
        uncontam_logden_vals = uncontam_sm.log_density(new_data, uncontam_coef, True)

        file_name_newdata = f'/new_data.npy'
        np.save(full_save_folder + file_name_newdata, new_data)

        file_name_coef = f'/logpenparam={log_pen_param}-uncontam-coef.npy'
        np.save(full_save_folder + file_name_coef, uncontam_coef)

        file_name_diff = f'/logpenparam={log_pen_param}-uncontam-logden-newdata.npy'
        np.save(full_save_folder + file_name_diff, uncontam_logden_vals)

        contam_sm = ContamSMFinKernelExpFam(
            data=data_waiting,
            contam_data=contam_data.reshape(-1, 1),
            grid_points=grid_points,
            contam_weight=contam_weight,
            penalty_param=np.exp(log_pen_param),
            base_density=base_density,
            r1=r1,
            r2=r2,
            c=c,
            bw=bw,
            kernel_type='gaussian_poly2')

        contam_coef = contam_sm.coef()
        contam_norm = np.sqrt(contam_coef.T @ gram_mat @ contam_coef).item()
        contam_logden_vals = contam_sm.log_density(new_data, contam_coef, True)

        IF_vals = (contam_logden_vals - uncontam_logden_vals) / contam_weight

        # save coefficients
        file_name_coef = f'/logpenparam={log_pen_param}-contam-coef.npy'
        np.save(full_save_folder + file_name_coef, contam_coef)

        file_name_diff = f'/logpenparam={log_pen_param}-contam-logden-newdata.npy'
        np.save(full_save_folder + file_name_diff, contam_logden_vals)

        file_name_ifvals = f'/logpenparam={log_pen_param}-IF-newdata.npy'
        np.save(full_save_folder + file_name_ifvals, IF_vals)

        print(uncontam_norm, contam_norm, np.max(np.abs(IF_vals)))
