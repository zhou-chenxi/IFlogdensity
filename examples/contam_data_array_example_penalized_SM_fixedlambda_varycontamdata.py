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
    contam_data_array = np.arange(5., 310., 5)

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 7.0
    
    # penalty parameter
    log_pen_param = -12.0
    
    print(f'bw={bw}, and penalty parameter=exp({log_pen_param}).')
    
    # contamnation weight
    contam_weight = 0.001
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 310.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_cnt = plot_kwargs['plot_pts_cnt']
    
    result = eval_IF_SMlogdensity_contam_data_array(
        data=data_waiting,
        new_data=np.linspace(plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1], plot_kwargs['plot_pts_cnt']),
        contam_data_array=contam_data_array,
        contam_weight=contam_weight,
        penalty_param=np.exp(log_pen_param),
        base_density=base_density,
        r1=1.0,
        r2=0.,
        c=0.,
        bw=bw,
        kernel_type=kernel_type,
        save_data=True,
        save_dir=f'PenSM-bw={bw}-kernel={kernel_type}-loglambda={log_pen_param}-contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_cnt}')
