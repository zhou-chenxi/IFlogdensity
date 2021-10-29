#!/usr/bin/env python3

import os
import copy
from dekef.base_density import *
from IFdensity.influence_function_contam_data_list import *
from IFdensity.influence_function import *

if __name__ == '__main__':
    
    os.chdir('/Users/chenxizhou/Dropbox/code_package/IFdensity')
    true_data = np.load('data/geyser.npy').astype(np.float64)
    df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
    
    # original data with 108.0 removed
    data_waiting = df[df != 108.0]
    
    # array of contaminated data
    contam_data_array = np.sort(np.unique(np.concatenate((data_waiting.flatten(), np.arange(90., 401., 2)))))

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 5.0
    
    # penalty parameter
    log_pen_param = -8.0
    
    print(f'bw={bw}, and penalty parameter=exp({log_pen_param}).')
    
    # contamnation weight
    contam_weight = 1e-8
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(21., 410.))
    
    new_data = np.linspace(
        plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1], plot_kwargs['plot_pts_cnt']).reshape(-1, 1)
    
    for contam_data in contam_data_array:

        print('-' * 50)
        print(f'Current contaminated data point is {contam_data}.')
    
        contam_data = np.array([[contam_data]])
        
        ifun = SMInfluenceFunction(
            data=data_waiting,
            contam_data=contam_data,
            contam_weight=1e-8,
            penalty_param=np.exp(log_pen_param),
            base_density=base_density,
            r1=1.0,
            r2=0.,
            c=0.,
            bw=bw,
            kernel_type='gaussian_poly2')
        
        ifun_natparam_file_name = f'/contam_data={contam_data[0]}-IF-natparam-newdata.npy'
        ifun_natparam = ifun.eval_IF_natparam(new_data=new_data)

        ifun_natparamlimit_file_name = f'/contam_data={contam_data[0]}-IF-natparam-limit-newdata.npy'
        ifun_natparam_limit = ifun.eval_IF_natparam_limit(new_data=new_data)
        
        save_dir = f'data/bw={bw}-kernel={kernel_type}-loglambda={log_pen_param}-contamweight={contam_weight}'
        np.save(save_dir + ifun_natparam_file_name, ifun_natparam)
        np.save(save_dir + ifun_natparamlimit_file_name, ifun_natparam_limit)
