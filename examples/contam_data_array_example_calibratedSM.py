#!/usr/bin/env python3

import os
import copy
from dekef.base_density import *
from IFlogdensity.influence_function_contam_data_list import *
from IFlogdensity.influence_function_sm_de import *
from IFlogdensity.calibrate_sm_ml import *

if __name__ == '__main__':
    
    os.chdir('/Users/chenxizhou/Dropbox/code_package/IFlogdensity')
    true_data = np.load('data/geyser.npy').astype(np.float64)
    df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
    
    # original data with 108.0 removed
    data_waiting = df[df != 108.0]
    
    # array of contaminated data
    contam_data_array = np.arange(2., 410., 4).reshape(-1, 1)
    # np.sort(np.unique(np.concatenate([np.arange(2., 410., 4), np.arange(40., 100., 2)])))
    # np.sort(np.unique(np.concatenate((np.arange(90., 401., 2), data_waiting.flatten()))))

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 7.0
    
    # penalty parameter
    log_pen_param_ml = -10.0
    
    print(f'bw={bw}, and penalty parameter=exp({log_pen_param_ml}).')
    
    # contamnation weight
    contam_weight = 0.01
    
    # seed number
    seed = 0
    print(f'current seed number = {seed}')
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 410.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']
    
    # uncontaminated density
    basisn = 205
    
    abstol = 0.05
    step_size = 0.6
    grid_points_ml = np.arange(1., 411., 2).reshape(-1, 1)
    # save_dir_ml = (f'/Users/chenxizhou/Dropbox/code_package/IFlogdensity/data/ML-basisn={basisn}-bw={bw}-
    #                kernel={kernel_type}-' +
    #                f'loglambda={log_pen_param_ml}-contamweight={contam_weight}-plotdomain={plot_xlimit}-' +
    #                f'plotcnts={plot_pts_cnt}-abstol={abstol}-stepsize={step_size}-seed={seed}')

    start_t_val = 10000.0
    save_dir_ml = (f'data/ConstrainedML-basisn={basisn}-bw={bw}-kernel={kernel_type}-loglambda={log_pen_param_ml}-'
                   f'contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_pts_cnt}-oriabstol={abstol}-'
                   f'oristepsize={step_size}-seed={seed}-start_t_val={start_t_val}')
    
    # read in the ML coefficient vector
    coef_ml = np.load(save_dir_ml + f'/uncontam-coef.npy')
    print(len(coef_ml))
    cali_mlsm = CalibrateMLSM(
        data=data_waiting,
        contam_data=contam_data_array[0],
        kernel_type=kernel_type,
        r1=1.,
        r2=0.,
        c=0.,
        bw=bw,
        contam_weight=0.,
        base_density=base_density,
        coef_ml=coef_ml.reshape(-1, 1),
        penalty_param_ml=np.exp(log_pen_param_ml),
        grid_points_ml=grid_points_ml)
    
    # coefficients
    lambda_sm, coef_sm = cali_mlsm.bisection_sm_pen_param(
        left_penalty_param=np.exp(-10.),
        right_penalty_param=np.exp(0.),
        tol_param=1e-4,
        max_iter=50)
    print(f'Uncontaminated penalty parameter: {lambda_sm}')
    
    save_dir_sm = (f'/Users/chenxizhou/Dropbox/code_package/IFlogdensity/data/CalibratedSM-ConstrainedML-' +
                   f'basisn={basisn}-bw={bw}-kernel={kernel_type}-' +
                   f'loglambda={log_pen_param_ml}-contamweight={contam_weight}-plotdomain={plot_xlimit}-' +
                   f'plotcnts={plot_pts_cnt}-oriabstol={abstol}-oristepsize={step_size}-seed={seed}')
    if not os.path.isdir(save_dir_sm):
        os.mkdir(save_dir_sm)
    np.save(save_dir_sm + '/uncontam-coef.npy', coef_sm)
    
    # log density
    uncontam_logdensity_vals = cali_mlsm.calibrated_sm_log_density(
        new_data=np.linspace(plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1], plot_kwargs['plot_pts_cnt']),
        coef=coef_sm,
        compute_base_density=False)
    np.save(save_dir_sm + '/uncontam-logden-newdata.npy', uncontam_logdensity_vals)
    
    for contam_data in contam_data_array:
    
        # print(f'Current contaminated data is {contam_data.item()}.')

        # read in the ML coefficient vector
        coef_ml = np.load(save_dir_ml + f'/contam_data={contam_data}-contam-coef.npy')
        cali_mlsm_contam = CalibrateMLSM(
            data=data_waiting,
            contam_data=contam_data,
            kernel_type=kernel_type,
            r1=1.,
            r2=0.,
            c=0.,
            bw=bw,
            contam_weight=contam_weight,
            base_density=base_density,
            coef_ml=coef_ml.reshape(-1, 1),
            penalty_param_ml=np.exp(log_pen_param_ml),
            grid_points_ml=grid_points_ml)

        # coefficients
        lambda_sm, coef_sm = cali_mlsm_contam.bisection_sm_pen_param(
            left_penalty_param=np.exp(-10.),
            right_penalty_param=np.exp(0.),
            tol_param=1e-4,
            max_iter=50)
        print(f'Contaminated obs = {contam_data.item()}, penalty parameter = {lambda_sm}')
        # np.save(save_dir_sm + f'/contam_data={contam_data}-contam-coef.npy', coef_sm)
        #
        # # log density
        # contam_logdensity_vals = cali_mlsm_contam.calibrated_sm_log_density(
        #     new_data=np.linspace(plot_kwargs['x_limit'][0], plot_kwargs['x_limit'][1], plot_kwargs['plot_pts_cnt']),
        #     coef=coef_sm,
        #     compute_base_density=False)
        # np.save(save_dir_sm + f'/contam_data={contam_data}-contam-logden-newdata.npy', contam_logdensity_vals)
        #
        # # compute influence function
        # inf_fun_sm = (contam_logdensity_vals.flatten() - uncontam_logdensity_vals.flatten()) / contam_weight
        # np.save(save_dir_sm + f'/contam_data={contam_data}-IF-logden-newdata.npy', inf_fun_sm)
