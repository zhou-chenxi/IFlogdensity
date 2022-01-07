#!/usr/bin/env python3

import os
import copy
from dekef.base_density import *
from IFlogdensity.influence_function_contam_data_list import *
from IFlogdensity.influence_function_ml_de import *

if __name__ == '__main__':
    
    os.chdir('/Users/chenxizhou/Dropbox/code_package/IFlogdensity')
    true_data = np.load('data/geyser.npy').astype(np.float64)
    df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
    
    # original data with 108.0 removed
    data_waiting = df[df != 108.0]
    
    # array of contaminated data
    contam_data_array = np.array([10., 20.,  30.,  40.,  50.,  60.,  70.,  80.,
                                  90., 100., 150., 200., 250., 300., 350., 400.]).reshape(-1, 1)

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 5.0
    
    # penalty parameter
    log_pen_param = -4.0
    
    print(f'bw={bw}, and penalty parameter=exp({log_pen_param}).')
    
    # contamnation weight
    contam_weight = 1e-2
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 410.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_cnt = plot_kwargs['plot_pts_cnt']

    # grid points
    start_grid_points = np.arange(1., 411., 2)
    print(f'{len(start_grid_points)} basis functions are used.')

    # batch Monte Carlo parameters
    bmc_params = batch_montecarlo_params(
        mc_batch_size=2000,
        mc_tol=1e-3)

    # gradient descent algorithm parameters
    gdalgo_params = negloglik_optalgoparams(
        start_pt=np.zeros((start_grid_points.shape[0], 1)),
        step_size=0.4,
        max_iter=100,
        rel_tol=1e-5,
        abs_tol=0.05)
    abstol = gdalgo_params['abs_tol']
    stepsize = gdalgo_params['step_size']
    random_seed_nums = [0]
    
    print(f"Step size = {stepsize}.")
    print(f"Absolute tolerance parameter = {abstol}.")

    for i in random_seed_nums:
        np.random.seed(i)
        print(f'random number = {i}')
        result = eval_IF_MLlogdensity_contam_data_array(
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
            basis_type='grid_points',
            optalgo_params=gdalgo_params,
            batchmc_params=bmc_params,
            grid_points=start_grid_points,
            step_size_discount_factor=None,
            max_set_grid_points=None,
            rel_tol_param=None,
            save_data=True,
            save_dir=f'PenML-FixContamData-VaryLambda-basisn={len(start_grid_points)}-bw={bw}-kernel={kernel_type}-loglambda={log_pen_param}-contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_cnt}-abstol={abstol}-stepsize={stepsize}-seed={i}',
            print_error=True)
