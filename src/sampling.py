import numpy as np
import pandas as pd

from utils import *
from distance import *

import math
import ot
from sklearn.cluster import KMeans
from scipy import stats

def sampling_wass(arr, 
                  n_sample,
                  verbose = False,
                  pre_selection = True,
                  post_selection = True,
                  ):
    '''
    Heuristic function to find the nearest sample s.t. Wasserstein distance
    '''
    
    unique_points, unique_counts = np.unique(arr, return_counts = True, axis = 0)
    selected_points = np.zeros((n_sample, arr.shape[1]))
    if verbose:
        print(selected_points.shape)
    points_to_delete_index, number_selected_points = [], 0
    if pre_selection:
        threshold_cluster_attraction = len(arr) / n_sample #math.ceil(len(arr) / n_sample)
        if verbose:
            print(f'the threshold is {threshold_cluster_attraction}')
        for index, count in enumerate(unique_counts):
          if count > threshold_cluster_attraction:
            if verbose:
                print(f'the number of this unique points is {count}')
            indexs = np.where((arr == unique_points[index]).sum(axis = 1) == arr.shape[1])[0]
            new_number_selected_points = number_selected_points + math.floor(count/threshold_cluster_attraction)
            #Select those points
            selected_points[number_selected_points : new_number_selected_points, :] = arr[indexs[0],:]
            #Remove the points already represented by the selected points
            points_to_delete_index += list(indexs[:(count - (count % math.ceil(threshold_cluster_attraction)))])
            if verbose:
                print(f'the number of points to delete is {len(points_to_delete_index)}')
            number_selected_points = new_number_selected_points
            if verbose:
                print(f'the number of selected points is {number_selected_points}')
                print()
    
    new_arr = np.delete(arr, points_to_delete_index, axis = 0)
    
    if post_selection:
        threshold_cluster_attraction = math.ceil(len(new_arr) / n_sample)
        if verbose:
            print(f'the threshold is {threshold_cluster_attraction}')
        
        kmeans = KMeansConstrained(
                 n_clusters= n_sample - number_selected_points,
                 size_min=threshold_cluster_attraction,
                 size_max=threshold_cluster_attraction + 1,
                 random_state=0
                ).fit(new_arr)
        centroid                    = kmeans.cluster_centers_
        Dist_matrix_points_centroid = ot.dist(x1 = centroid, x2 = new_arr, p=2, metric = 'euclidean')
        selected_index              = np.argmin(Dist_matrix_points_centroid, axis = 1)
        
        '''kmeans = KMeans(n_clusters=n_sample - number_selected_points,
                        max_iter = 5000,
                        init = 'random',
                        n_init = 50,
                        tol = 1e-8,
                      ).fit(new_arr)
        centroid                    = kmeans.cluster_centers_
        Dist_matrix_points_centroid = ot.dist(x1 = centroid, x2 = new_arr, p=2, metric = 'euclidean')
        selected_index              = np.argmin(Dist_matrix_points_centroid, axis = 1)'''
    else:
        selected_index              = np.random.choice(np.arange(len(new_arr)), replace=False, size=n_sample - number_selected_points)
    selected_points[number_selected_points:,:] = new_arr[selected_index]

    return selected_points
    
def sampling_wass_to_other(arr_sampled,
                           arr_goal,
                           n_sample,
                           verbose = False,
                           pre_selection = True,
                           post_selection = True,
                           ):
    '''
    Heuristic function to find the nearest sample s.t. Wasserstein distance
    '''
    
    unique_points, unique_counts = np.unique(arr_goal, 
                                             return_counts = True, 
                                             axis = 0)
    selected_points = np.zeros((n_sample, arr_sampled.shape[1]))
    if verbose:
        print(selected_points.shape)
    points_to_delete_index, number_selected_points = [], 0
    if pre_selection:
        threshold_cluster_attraction = len(arr_sampled) / n_sample #math.ceil(len(arr) / n_sample)
        if verbose:
            print(f'the threshold is {threshold_cluster_attraction}')
        for index, count in enumerate(unique_counts):
          if count > threshold_cluster_attraction:
            indexs = np.where((arr_sampled == unique_points[index]).sum(axis = 1) == arr_sampled.shape[1])[0]
            num_indexs = len(indexs)
            if verbose:
                print(f'the number of this unique points is {count}, there is {num_indexs} in the sampled set')
            #We only pre-select if 
            if num_indexs > 0:
                max_selected = math.floor(count/threshold_cluster_attraction)
                num_selected = min(max_selected, num_indexs)
                new_number_selected_points = number_selected_points + num_selected 
                #Select those points
                selected_points[number_selected_points : new_number_selected_points, :] = arr_sampled[indexs[0],:]
                #Remove the points already represented by the selected points
                points_to_delete_index += list(indexs[:num_selected])
                if verbose:
                    print(f'the number of points to delete is {len(points_to_delete_index)}')
                number_selected_points = new_number_selected_points
                if verbose:
                    print(f'the number of selected points is {number_selected_points}')
                    print()
    
    new_arr_goal = np.delete(arr_goal, points_to_delete_index, axis = 0)
    
    if post_selection:
        kmeans = KMeans(n_clusters=n_sample - number_selected_points,
                        max_iter = 5000,
                        init = 'random',
                        n_init = 50,
                        tol = 1e-8,
                      ).fit(new_arr_goal)
        centroid                    = kmeans.cluster_centers_
        Dist_matrix_points_centroid = ot.dist(x1 = centroid, 
                                              x2 = new_arr_goal, 
                                              p=2, 
                                              metric = 'euclidean')
        selected_index              = np.argmin(Dist_matrix_points_centroid, axis = 1)
    else:
        selected_index              = np.random.choice(np.arange(len(new_arr_goal)),
                                                       replace=False,
                                                       size= n_sample - number_selected_points)
    selected_points[number_selected_points:,:] = new_arr_goal[selected_index]

    return selected_points
    
    
def sampling_wass_DI(arr, 
                     n_sample,
                     S_col = None, 
                     Y_col = None,
                     verbose = False,
                     pre_selection = True,
                     post_selection = True,
                     ):
    '''
    Heuristic function to find the nearest sample s.t. Wasserstein distance, while keeping the same DI
    '''
    
    unique_points, unique_counts = np.unique(arr, 
                                             return_counts = True, 
                                             axis = 0)
    selected_points = np.zeros((n_sample, arr.shape[1]))
    if verbose:
        print(selected_points.shape)
    points_to_delete_index, number_selected_points = [], 0
    if pre_selection:
        threshold_cluster_attraction = len(arr) / n_sample #math.ceil(len(arr) / n_sample)
        if verbose:
            print(f'the threshold is {threshold_cluster_attraction}')
        for index, count in enumerate(unique_counts):
          if count > threshold_cluster_attraction:
            if verbose:
                print(f'the number of this unique points is {count}')
            indexs = np.where((arr == unique_points[index]).sum(axis = 1) == arr.shape[1])[0]
            new_number_selected_points = number_selected_points + math.floor(count/threshold_cluster_attraction)
            #Select those points
            selected_points[number_selected_points : new_number_selected_points, :] = arr[indexs[0],:]
            #Remove the points already represented by the selected points
            points_to_delete_index += list(indexs[:(count - (count % math.ceil(threshold_cluster_attraction)))])
            if verbose:
                print(f'the number of points to delete is {len(points_to_delete_index)}')
            number_selected_points = new_number_selected_points
            if verbose:
                print(f'the number of selected points is {number_selected_points}')
                print()
    
    arr = np.delete(arr, points_to_delete_index, axis = 0)
    
    if not ( (S_col is None) or (Y_col is None) ):
        S_index, Y_index = arr[:,S_col] == 1, arr[:,Y_col] == 1
        list_X = np.array([arr[S_index * Y_index,:], 
                           arr[S_index * (1-Y_index),:], 
                           arr[(1-S_index) * Y_index,:],
                           arr[(1-S_index) * (1-Y_index),:]
                           ])
    else:
        list_X = np.array([arr])
    n_cluster_bins = np.zeros(len(list_X))
    coef_number_sample = n_sample / len(arr)
    for count, array in enumerate(list_X):
        n_cluster_bins[count] = math.floor(len(array) * coef_number_sample)
    #Will only happen if S_col & Y_col are not None
    while n_cluster_bins.sum < len(arr):
        #We prefer, because we aim to follow the DI constraint, to add S0Y1 individual ; 
        # adding S1Y0 ind would have been ok, but we tend to have more S1 than S0.
        n_cluster_bins[2] +=1
    former_count = number_selected_points
    for count, array in enumerate(list_X):
        if post_selection:
            kmeans = KMeans(n_clusters=n_cluster_bins[count],
                            random_state=0, 
                            n_init="auto",
                            ).fit(array)
            centroid = kmeans.cluster_centers_
            Dist_matrix_points_centroid = ot.dist(x1 = centroid, 
                                                  x2 = array, 
                                                  p=2, 
                                                  metric = 'euclidean')
            selected_index  = np.argmin(Dist_matrix_points_centroid, axis = 0)
            
        else:
            selected_index = np.random.choice(np.arange(len(array)), 
                                              replace=False,
                                              size=n_sample - number_selected_points)
        selected_points[former_count : (former_count + n_cluster_bins[count]),:] = array[selected_index]
        former_count = former_count + n_cluster_bins[count]
        
    return selected_points

def sample_arr(array, 
               n_sample):
    indexs = np.arange(len(array))
    np.random.shuffle(indexs)
    return array[indexs[:n_sample]]

def sample_arr_unique(array, 
                      n_sample):
    unique_points = np.unique(array, axis = 0)
    assert len(unique_points) > n_sample, 'the choice of sampling is surprising giving the amount of repeated data or the low sample number'
    indexs = np.arange(len(unique_points))
    np.random.shuffle(indexs)
    return unique_points[indexs[:n_sample]]

def find_W_KL_cdt(arr_P,
                  n_sample,
                  iter_precision,
                  max_wass_iter = 1e7,
                  verbose = False,
                  only_KL = False,
                  only_W  = False,
                  ):
    assert not (only_KL and only_W), 'the function would do nothing with those parameters : please have at least either only_KL or only_W equal False'
    arr_KL, arr_W, arr_DI = np.zeros(iter_precision), np.zeros(iter_precision), np.zeros(iter_precision)
    for i in range(iter_precision):
        sample = sample_arr(arr_P, n_sample)
        
        if not only_W:
            arr_KL[i] = KL_arr(arr_P = arr_P,
                            arr_Q = sample)
        
        if not only_KL:
            arr_W[i]  = Wasserstein(arr_P, 
                                sample, 
                                numItermax=max_wass_iter)
        
        arr_DI[i] = DI_fct(sample, 
                           S_index = -2, 
                           Y_index = -1)

    mean_KL, std_KL = arr_KL.mean(), arr_KL.std()
    mean_W , std_W = arr_W.mean(),  arr_W.std()
    mean_DI, std_DI = arr_DI.mean(), arr_DI.std()

    KL_cdt, W_cdt = mean_KL + 1.96*std_KL, mean_W + 1.96*std_W
    if verbose:
        if not only_KL:
            print(f'Wasserstein expected is in [{np.round((mean_W - 1.96*std_W), 4)}, {np.round((mean_W + 1.96*std_W), 4)}]')
        if not only_W:
            print(f'Kullback Leibler expected is in [{np.round((mean_KL - 1.96*std_KL), 4)}, {np.round((mean_KL + 1.96*std_KL), 4)}]')
        print(f'Disparate impact expected with random sample is in [{np.round((mean_DI - 1.96*std_DI), 4)}, {np.round((mean_DI + 1.96*std_DI), 4)}]')
    return W_cdt, KL_cdt

def sample_arr_unique_DI(array, 
                         n_sample,
                         S_index = -2,
                         Y_index = -1,
                         verbose = False,
                         ):
    selected_points = np.zeros((n_sample, array.shape[1]))
    S1_index = array[:,S_index] == 1
    Y1_index = array[:,Y_index] == 1

    # ~ reverse the index
    Y1S1_index, Y0S1_index, Y1S0_index, Y0S0_index =  S1_index & Y1_index ,S1_index & (~Y1_index),(~S1_index) & Y1_index ,(~S1_index) & (~Y1_index)

    #Next line just for clarity
    Y1S1, Y0S1, Y1S0, Y0S0 = array[Y1S1_index], array[Y0S1_index], array[Y1S0_index], array[Y0S0_index]
    bins = np.array([len(Y1S1), len(Y0S1), len(Y1S0), len(Y0S0)], dtype = int)
    coef_reduction = len(array) / n_sample
    new_bins       = np.array(bins // coef_reduction, dtype = int)

    while new_bins.sum() < n_sample:
        new_bins[2] +=1
        

    keep_count = 0
    for count, bin_arr in enumerate([Y1S1, Y0S1, Y1S0, Y0S0]):
        unique_points = np.unique(bin_arr, axis = 0)
        if verbose:
            print(f'S mean : {unique_points[:,S_index].mean()}, Y/Pred mean {unique_points[:,Y_index].mean()}')
        len_u = len(unique_points)
        indexs = np.arange(len_u)
        np.random.shuffle(indexs)
        bin_count = 0
        while bin_count < new_bins[count]:
            num_selected_points_first = min(len_u, new_bins[count] - bin_count)
            if verbose:
                print(f'number selected points : {num_selected_points_first}')
            selected_points[keep_count:(keep_count + num_selected_points_first),:] = unique_points[indexs[:num_selected_points_first], :]

            keep_count += num_selected_points_first
            bin_count += num_selected_points_first
            if verbose:
                print(f'total selected points : {(abs(selected_points).sum(axis = 1) != 0).sum()}')

    return selected_points