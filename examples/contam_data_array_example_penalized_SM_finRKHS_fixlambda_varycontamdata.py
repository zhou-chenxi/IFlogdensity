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
    contam_data_list = np.arange(5., 310., 5)
    
    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 9.0
    
    # penalty parameter
    log_pen_param = -10.
    
    print(f'bw={bw}, log penalty parameter = {log_pen_param}.')
    
    # contamnation weight
    contam_weight = 1e-3
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 310.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']

    new_data = np.linspace(plot_xlimit[0], plot_xlimit[1], plot_pts_cnt)

    full_save_folder = (f'data/PenSM-FinKEF-bw={bw}-kernel={kernel_type}-logpenparam={log_pen_param}-' +
                        f'contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_pts_cnt}')
    if not os.path.isdir(full_save_folder):
        os.mkdir(full_save_folder)
        
    r1, r2, c = 1., 0., 0.
    grid_points = np.arange(1., 311., 1)

    uncontam_sm = ContamSMFinKernelExpFam(
        data=data_waiting,
        contam_data=contam_data_list[0].reshape(-1, 1),
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

    uncontam_logden_vals = uncontam_sm.log_density(new_data, uncontam_coef, True)

    file_name_newdata = f'/new_data.npy'
    np.save(full_save_folder + file_name_newdata, new_data)

    file_name_coef = f'/uncontam-coef.npy'
    np.save(full_save_folder + file_name_coef, uncontam_coef)

    file_name_diff = f'/uncontam-logden-newdata.npy'
    np.save(full_save_folder + file_name_diff, uncontam_logden_vals)

    for contam_data in contam_data_list:
        
        print('*' * 50)
        print(f'contam data = {contam_data}')

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
        contam_logden_vals = contam_sm.log_density(new_data, contam_coef, True)

        IF_vals = (contam_logden_vals - uncontam_logden_vals) / contam_weight

        # save coefficients
        file_name_coef = f'/contam_data={contam_data}-contam-coef.npy'
        np.save(full_save_folder + file_name_coef, contam_coef)

        file_name_diff = f'/contam_data={contam_data}-contam-logden-newdata.npy'
        np.save(full_save_folder + file_name_diff, contam_logden_vals)

        file_name_ifvals = f'/contam_data={contam_data}-IF-newdata.npy'
        np.save(full_save_folder + file_name_ifvals, IF_vals)
