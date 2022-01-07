#!/usr/bin/env python3

import os
import copy
from dekef.base_density import *
from IFlogdensity.influence_function_contam_data_list import *
from IFlogdensity.influence_function_sm_de import *

if __name__ == '__main__':
    
    os.chdir('/Users/chenxizhou/Dropbox/code_package/IFlogdensity')
    true_data = np.load('data/geyser.npy').astype(np.float64)
    df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
    
    # original data with 108.0 removed
    data_waiting = df[df != 108.0]
    
    # array of contaminated data
    contam_data_array = np.array([10., 20.,  30.,  40.,  50.,  60.,  70.,  80.,
                                  90., 100., 150., 200., 250., 300., 350., 400.]).reshape(-1, 1)
    # np.sort(np.unique(np.concatenate([np.arange(2., 410., 4), np.arange(40., 100., 2)]))).reshape(-1, 1)
    # np.sort(np.unique(np.concatenate((np.arange(90., 401., 2), data_waiting.flatten()))))

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # parameters with kernel function
    r1 = 1.0
    r2 = 0.0
    c = 0.0
    bw = 9.0
    
    # penalty parameter
    log_pen_param_list = np.arange(-13., 1.)
    
    print(f'bw={bw}')
    
    # contamnation weight
    contam_weight = 0.01
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 410.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_cnt = plot_kwargs['plot_pts_cnt']

    for log_pen_param in log_pen_param_list:
        
        print(f'Current penalty parameter = {log_pen_param}')
    
        save_dir = f'SM-FixContamData-VaryLambda-bw={bw}-kernel={kernel_type}-loglambda={log_pen_param}-' \
                   f'contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_cnt}'
        
        print('compute the coefficients of uncontaminated density estimates. ')
        uncontam_density = ContamSMDensityEstimate(
            data=data_waiting,
            contam_data=contam_data_array[0].reshape(-1, 1),
            contam_weight=0.,
            penalty_param=np.exp(log_pen_param),
            base_density=base_density,
            r1=r1,
            r2=r2,
            c=c,
            bw=bw,
            kernel_type=kernel_type)
    
        coef = uncontam_density.coef()
    
        np.save('data/' + save_dir + f'/uncontam-coef.npy', coef)
    
        for contam_data in contam_data_array:
    
            print(f'compute the coefficients of contaminated density estimate with observation = {contam_data}. ')
    
            contam_density = ContamSMDensityEstimate(
                data=data_waiting,
                contam_data=contam_data,
                contam_weight=contam_weight,
                penalty_param=np.exp(log_pen_param),
                base_density=base_density,
                r1=r1,
                r2=r2,
                c=c,
                bw=bw,
                kernel_type=kernel_type)
            
            coef = contam_density.coef()
            
            np.save('data/' + save_dir + f'/contam_data={contam_data}-contam-coef.npy', coef)