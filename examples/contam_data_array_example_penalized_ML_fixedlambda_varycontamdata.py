#!/usr/bin/env python3

import os
import copy
from dekef.base_density import *
from IFlogdensity.influence_function_contam_data_list import *
from IFlogdensity.influence_function_ml_de import *
from datetime import datetime as dt


if __name__ == '__main__':
    
    os.chdir('/Users/chenxizhou/Dropbox/code_package/IFlogdensity')
    true_data = np.load('data/geyser.npy').astype(np.float64)
    df = copy.deepcopy(true_data[:, 0]).reshape(-1, 1)
    
    # original data with 108.0 removed
    data_waiting = df[df != 108.0]
    
    # array of contaminated data
    contam_data_array = np.arange(0., 270., 5)

    # kernel function used
    kernel_type = 'gaussian_poly2'

    # bandwidth parameter in the Gaussian kernel function
    bw = 9.0
    
    # penalty parameter
    log_pen_param = -8.0
    
    print(f'bw={bw}, and penalty parameter=exp({log_pen_param}).')
    
    # contamnation weight
    contam_weight = 1e-3
    
    # base density
    base_density = BasedenGamma(np.load('data/geyser.npy').astype(np.float64)[:, 0])
    plot_kwargs = plot_IF_1d_params(x_limit=(1., 310.), plot_pts_cnt=3000)
    plot_xlimit = plot_kwargs['x_limit']
    plot_pts_cnt = plot_kwargs['plot_pts_cnt']

    # grid points
    start_grid_points = np.arange(1., 311., 1)
    print(f'{len(start_grid_points)} basis functions are used.')

    # batch Monte Carlo parameters
    bmc_params = batch_montecarlo_params(
        mc_batch_size=5000,
        mc_tol=1e-3)

    # gradient descent algorithm parameters
    gdalgo_params = negloglik_optalgoparams(
        start_pt=np.zeros((start_grid_points.shape[0], 1)),
        step_size=0.3,
        max_iter=100,
        rel_tol=1e-5,
        abs_tol=0.05)
    abstol = gdalgo_params['abs_tol']
    stepsize = gdalgo_params['step_size']
    random_seed_nums = [1]
    
    print(f"Step size = {stepsize}.")
    print(f"Absolute tolerance parameter = {abstol}.")

    r1, r2, c = 1., 0., 0.
    kernel_function_grid = GaussianPoly2(
        data=start_grid_points.reshape(-1, 1),
        r1=r1,
        r2=r2,
        c=c,
        bw=bw)
    gram_grid = kernel_function_grid.kernel_gram_matrix(start_grid_points.reshape(-1, 1))

    for seed_number in random_seed_nums:

        uncontam_ml = ContamMLDensityEstimate(
            data=data_waiting,
            contam_data=np.array([0.]).reshape(-1, 1),
            contam_weight=0.,
            penalty_param=np.exp(log_pen_param),
            base_density=base_density,
            r1=r1,
            r2=r2,
            c=c,
            bw=bw,
            kernel_type='gaussian_poly2')

        np.random.seed(seed_number)
        print(f'start time = {dt.now().strftime("%H:%M:%S")}')
        uncontam_coef = uncontam_ml.coef_grid_points(
            optalgo_params=gdalgo_params,
            batchmc_params=bmc_params,
            algo='gd',
            step_size_factor=1.,
            grid_points=start_grid_points,
            print_error=True)
        print(f'end time = {dt.now().strftime("%H:%M:%S")}')
        
        uncontam_norm = np.sqrt(uncontam_coef[0].T @ gram_grid @ uncontam_coef[0]).item()

        new_data = np.linspace(plot_xlimit[0], plot_xlimit[1], plot_pts_cnt)
        uncontam_logden_vals = uncontam_ml.log_density(new_data, uncontam_coef, True)

        full_save_folder = (
                    f'data/PenML-GD-basisn={len(start_grid_points)}-bw={bw}-kernel={kernel_type}-'
                    f'log_pen_param={log_pen_param}-' +
                    f'contamweight={contam_weight}-plotdomain={plot_xlimit}-plotcnts={plot_pts_cnt}-' +
                    f'seed={seed_number}')
        if not os.path.isdir(full_save_folder):
            os.mkdir(full_save_folder)
       
        file_name_newdata = f'/new_data.npy'
        np.save(full_save_folder + file_name_newdata, new_data)

        file_name_grid_points = f'/grid_points.npy'
        np.save(full_save_folder + file_name_grid_points, start_grid_points)

        file_name_coef = f'/uncontam-coef.npy'
        np.save(full_save_folder + file_name_coef, uncontam_coef[0])

        file_name_diff = f'/uncontam-logden-newdata.npy'
        np.save(full_save_folder + file_name_diff, uncontam_logden_vals)
        
        for contam_data in contam_data_array:
            
            print('-' * 50)
            print(f'Current contaminated observation = {contam_data}')

            contam_ml = ContamMLDensityEstimate(
                data=data_waiting,
                contam_data=np.array(contam_data).reshape(-1, 1),
                contam_weight=contam_weight,
                penalty_param=np.exp(log_pen_param),
                base_density=base_density,
                r1=r1,
                r2=r2,
                c=c,
                bw=bw,
                kernel_type='gaussian_poly2')

            np.random.seed(seed_number)
            print(f'start time = {dt.now().strftime("%H:%M:%S")}')
            contam_coef = contam_ml.coef_grid_points(
                optalgo_params=gdalgo_params,
                batchmc_params=bmc_params,
                algo='gd',
                step_size_factor=1.,
                grid_points=start_grid_points,
                print_error=True)
            print(f'end time = {dt.now().strftime("%H:%M:%S")}')

            contam_norm = np.sqrt(contam_coef[0].T @ gram_grid @ contam_coef[0]).item()

            contam_logden_vals = contam_ml.log_density(new_data, contam_coef, True)

            IF_vals = (contam_logden_vals - uncontam_logden_vals) / contam_weight

            # save coefficients
            file_name_coef = f'/contam_data={contam_data}-contam-coef.npy'
            np.save(full_save_folder + file_name_coef, contam_coef[0])

            file_name_diff = f'/contam_data={contam_data}-contam-logden-newdata.npy'
            np.save(full_save_folder + file_name_diff, contam_logden_vals)

            file_name_ifvals = f'/contam_data={contam_data}-IF-newdata.npy'
            np.save(full_save_folder + file_name_ifvals, IF_vals)
            
            print(f'contaminated data = {contam_data}, uncontam_f_norm={uncontam_norm}, contam_f_norm={contam_norm}, '
                  f'IF sup norm = {np.max(np.abs(IF_vals))}')
