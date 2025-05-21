import numpy as np
import scipy.stats as stats
from scipy.special import kl_div
import ot
import math
from utils import *

##############
#----KL------#
##############

def KL_arr(arr_P, 
           arr_Q,
           verbose = False):
    unique_P, counts_P = np.unique(arr_P, return_counts=True, axis = 0)
    unique_Q, counts_Q = np.unique(arr_Q, return_counts=True, axis = 0)

    prob_P, prob_Q = counts_P / counts_P.sum(), counts_Q / counts_Q.sum()
        
    dic_P = {str(unique_P[i]): prob_P[i] for i in range(len(unique_P))}
    dic_Q = {str(unique_Q[i]): prob_Q[i] for i in range(len(unique_Q))}

    bool_Q_in_P = True
    keys_P, keys_Q = list(dic_P.keys()), list(dic_Q.keys())
    for key in keys_Q:
        if key not in keys_P:
            bool_Q_in_P = False
            break
    if not bool_Q_in_P:
        if verbose:
            print('P is the true distribution : the elements of Q needs to be within P')
        return float('+inf')

    if verbose:
        print(dic_P, dic_Q)
        
    arr_P_intersection = np.array([dic_P[key] for key in keys_Q])
    
    assert len(arr_P_intersection) == len(prob_Q), 'error'
    
    KL = 0
    for i in range(len(arr_P_intersection)):
        KL += kl_div(arr_P_intersection[i], prob_Q[i])
    return KL

def KL_lambda_fct(lambdas):
    return np.sum(np.multiply(np.log(lambdas * len(lambdas)), lambdas))

def transform_arr_dic_like_KL(arr_1, arr_2,
                              verbose = False,
                              intersection = True):
    assert (arr_1.shape[1]<4) or (arr_2.shape[1]<4)

    unique_1, counts_1 = np.unique(arr_1, return_counts=True, axis = 0)
    unique_2, counts_2 = np.unique(arr_2, return_counts=True, axis = 0)

    if unique_1.shape[1] > 2:
        prob_1 = counts_1 * unique_1[:,2]
        unique_1 = unique_1[:,:2]
    else:
        prob_1 = counts_1/np.sum(counts_1)

    if unique_2.shape[1] > 2:
        prob_2 = counts_2 * unique_2[:,2]
        unique_2 = unique_2[:,:2]
    else:
        prob_2 = counts_2/np.sum(counts_2)
        
    dic_1 = {str(unique_1[i]): prob_1[i] for i in range(len(unique_1))}
    dic_2 = {str(unique_2[i]): prob_2[i] for i in range(len(unique_2))}

    if verbose:
        print(dic_1, dic_2)
        
    if intersection:
        keys = list(set(list(dic_1.keys())) & set(list(dic_2.keys())))
        arr_1 = np.array([dic_1[key] for key in keys])
        arr_2 = np.array([dic_2[key] for key in keys])
    else:
        keys = list(set(list(dic_1.keys())).union(set(list(dic_2.keys()))))
        arr_1, arr_2 = [], []
        for key in keys:
            try:
                arr_1.append(dic_1[key])
            except:
                arr_1.append(0)
            try:
                arr_2.append(dic_2[key])
            except:
                arr_2.append(0)               
        arr_1, arr_2 = np.array(arr_1), np.array(arr_2)
    return arr_1, arr_2

def KL_fct(ars):
    KL = 0
    arr_1, arr_2 = ars[0], ars[1]
    for i in range(len(arr_1)):
        KL += kl_div(arr_1[i], arr_2[i])
    return KL

def KL_multidim_gaussian_approx(arr_p, arr_q):
    k = arr_p.shape[1]
    μ_p = arr_p.mean(0)
    μ_q = arr_q.mean(0)
    Σ_p = np.cov(arr_p, rowvar=False)
    Σ_q = np.cov(arr_q, rowvar=False)
    Σ_q_1 = np.linalg.inv(Σ_q) 
    #print(μ_p.shape, μ_q.shape, Σ_q.shape, Σ_p.shape, Σ_q_1.shape)
    return 1/2 * (  np.log(np.linalg.det(Σ_q)/np.linalg.det(Σ_p)) - k +(μ_p - μ_q).transpose() @ Σ_q_1 @ (μ_p - μ_q) + np.diag(Σ_q_1 @ Σ_p).sum() )

##############
#-----W-----#
##############

def Wasserstein_1D_same_size(arr_1, arr_2):
    '''
    In one dimension, the Wasserstein is only looking at the quantile (because (x+y)^2 > x^2 + y^2)
    '''
    assert arr_1.shape == arr_2.shape, 'the array are not the same shape'
    return ((np.sort(arr_1) - np.sort(arr_2))**2).mean()

def Wasserstein_1D_weighted(arr, 
                            weights_1,
                            weights_2,
                            verbose = False,
                            leftover_threshold = 1e-7):
    wass_dist = 0
    weights_1, weights_2 = weights_1 / weights_1.sum(), weights_2 / weights_2.sum()
    
    index = np.argsort(arr)
    arr, weights_1, weights_2 = arr[index], weights_1[index], weights_2[index]
    
    unique, unique_indices = np.unique(arr, return_index= True)
    arr_1, arr_2 = np.zeros((len(unique),2)), np.zeros((len(unique),2))
    if verbose:
        print(unique)

    for ind, u in enumerate(unique):
        try:
            below, above = unique_indices[ind], unique_indices[ind+1]
        except:
            below, above = unique_indices[ind],len(arr)
        if verbose:
            print(u, below,above)
        arr_1[ind, :] = [u, weights_1[below : above].sum()]
        arr_2[ind, :] = [u, weights_2[below : above].sum()]
    if verbose:
        print(arr_1, arr_2)

    arr, weights_1, weights_2 = arr_1[:,0], arr_1[:,1], arr_2[:,1]
    arr_1, arr_2 = arr.copy(), arr.copy()    
    aw_arr_1, aw_arr_2 = weights_1[0], weights_2[0]

    i_1, i_2 = 0, 0
    while (i_1 < len(arr_1)) and (i_2 < len(arr_2)):
        if verbose:
            print(aw_arr_1, aw_arr_2, arr_1[i_1], arr_2[i_2], (arr_1[i_1] - arr_2[i_2])**2)
        if aw_arr_1 > aw_arr_2:
            wass_dist += aw_arr_2 * (arr_1[i_1] - arr_2[i_2])**2
            aw_arr_1 -= aw_arr_2
            i_2 += 1
            try:
                aw_arr_2 = weights_2[i_2]
            except:
                aw_arr_2 = 0
            if verbose:
                print(f'i_2 is equal to {i_2}')
        elif aw_arr_1 < aw_arr_2:
            wass_dist += aw_arr_1 * (arr_1[i_1] - arr_2[i_2])**2
            aw_arr_2 -= aw_arr_1
            i_1 += 1
            try:
                aw_arr_1 = weights_1[i_1]
            except:
                aw_arr_1 = 0
            if verbose:
                print(f'i_1 is equal to {i_1}')
        else:
            #the two weights are equal : aw_arr_1 = aw_arr_2
            wass_dist += aw_arr_2 * (arr_1[i_1] - arr_2[i_2])**2
            i_1 += 1
            i_2 += 1
            try:
                aw_arr_1 = weights_1[i_1]
                aw_arr_2 = weights_2[i_2]
            except:
                aw_arr_1 = 0
                aw_arr_2 = 0
            if verbose:
                print(f'i_1 is equal to {i_1}')
                print(f'i_2 is equal to {i_2}')
        if verbose:
            print(wass_dist)
        
    assert max(aw_arr_1, aw_arr_2) < leftover_threshold, f'the value left are above the designed treshold : values are {aw_arr_1}, and {aw_arr_2}'
    return wass_dist

def Wasserstein_1D_same_size_weighted(arr_1, weights_1, arr_2, weights_2,
                                      verbose = False,
                                      leftover_threshold = 1e-7):
    weights_1, weights_2 = weights_1 / weights_1.sum(), weights_2 / weights_2.sum()
    index_1, index_2 = np.argsort(arr_1), np.argsort(arr_2)
    arr_1, weights_1 = arr_1[index_1], weights_1[index_1]
    arr_2, weights_2 = arr_2[index_2], weights_2[index_2]
    wass_dist = 0
    aw_arr_1, aw_arr_2 = weights_1[0], weights_2[0]
    i_1, i_2 = 0, 0
    while (i_1 < len(arr_1)) and (i_2 < len(arr_2)):
        if verbose:
            print(aw_arr_1, aw_arr_2, arr_1[i_1], arr_2[i_2], (arr_1[i_1] - arr_2[i_2])**2)
        if aw_arr_1 > aw_arr_2:
            wass_dist += aw_arr_2 * (arr_1[i_1] - arr_2[i_2])**2
            aw_arr_1 -= aw_arr_2
            i_2 += 1
            try:
                aw_arr_2 = weights_2[i_2]
            except:
                aw_arr_2 = 0
            if verbose:
                print(f'i_2 is equal to {i_2}')
        elif aw_arr_1 < aw_arr_2:
            wass_dist += aw_arr_1 * (arr_1[i_1] - arr_2[i_2])**2
            aw_arr_2 -= aw_arr_1
            i_1 += 1
            try:
                aw_arr_1 = weights_1[i_1]
            except:
                aw_arr_1 = 0
            if verbose:
                print(f'i_1 is equal to {i_1}')
        else:
            #the two weights are equal : aw_arr_1 = aw_arr_2
            wass_dist += aw_arr_2 * (arr_1[i_1] - arr_2[i_2])**2
            i_1 += 1
            i_2 += 1
            try:
                aw_arr_1 = weights_1[i_1]
                aw_arr_2 = weights_2[i_2]
            except:
                aw_arr_1 = 0
                aw_arr_2 = 0
            if verbose:
                print(f'i_1 is equal to {i_1}')
                print(f'i_2 is equal to {i_2}')
        if verbose:
            print(wass_dist)
        
    assert max(aw_arr_1, aw_arr_2) < leftover_threshold, f'the value left are above the designed treshold : values are {aw_arr_1}, and {aw_arr_2}'
    return wass_dist

def Wasserstein(array_1, array_2, numItermax=10000):
    n1, n2 = len(array_1), len(array_2)
    M = ot.dist(x1 = array_1, x2 = array_2, p=2, metric = 'euclidean')
    return ot.emd2(np.ones(n1)/n1, np.ones(n2)/n2, M, numItermax=numItermax)

def Wasserstein_multidim_gaussian_approx(arr_p, arr_q):
    μ_p = arr_p.mean(0)
    μ_q = arr_q.mean(0)
    Σ_p = np.cov(arr_p, rowvar=False)
    Σ_q = np.cov(arr_q, rowvar=False)
    Σ_p_1_2 = matrix_square_root(Σ_p)
    #print(((μ_p - μ_q)**2).sum())
    #print(np.trace(Σ_p + Σ_q - 2*np.sqrt(Σ_p_1_2 @ Σ_q @ Σ_p_1_2)))
    return ((μ_p - μ_q)**2).sum() + np.trace(Σ_p + Σ_q - 2*matrix_square_root(Σ_p_1_2 @ Σ_q @ Σ_p_1_2))

def Wasserstein_swap(arr_1, arr_2):
    sign = lambda x: math.copysign(1, x)
    
    D_11 = np.sum(arr_1[:,0] * arr_1[:,1]) - np.sum(arr_2[:,0] * arr_2[:,1])
    D_10 = np.sum(arr_1[:,0] * (1 - arr_1[:,1])) - np.sum(arr_2[:,0] * (1 - arr_2[:,1]))
    D_01 = np.sum((1 - arr_1[:,0]) * arr_1[:,1]) - np.sum((1 - arr_2[:,0]) * arr_2[:,1])
    D_00 = np.sum((1 - arr_1[:,0]) * (1 - arr_1[:,1])) - np.sum((1 - arr_2[:,0]) * (1 - arr_2[:,1]))
    N = len(arr_1)
    cost = 0

    #Cost matrix is 0 in diag, sqrt(2) inv diag and 1 elsewhere, hence we do as much as possible on near points (1 cost instead of sqrt(2))
    if sign(D_00) != sign(D_01):
        if (abs(D_00) < abs(D_01)):
            cost += abs(D_00)
            D_01 += D_00
            D_00 = 0
        else:
            cost += abs(D_01)
            D_00 += D_01
            D_01 = 0
    #print(cost, D_00, D_01, D_10, D_11)

    if (sign(D_00) != sign(D_10)) and (D_00 != 0):
        if (abs(D_00) < abs(D_10)):
            cost += abs(D_00)
            D_10 += D_00
            D_00 = 0
        else:
            cost += abs(D_10)
            D_00 += D_10
            D_10 = 0
    #print(cost, D_00, D_01, D_10, D_11)

    if sign(D_11) != sign(D_10) and (D_10 != 0):
        if abs(D_11) < abs(D_10):
            cost += abs(D_11)
            D_10 += D_11
            D_11 = 0
        else:
            cost += abs(D_10)
            D_11 += D_10
            D_10 = 0
    #print(cost, D_00, D_01, D_10, D_11)

    if (sign(D_11) != sign(D_01)) and (D_01 != 0) and (D_11 != 0):
        if abs(D_11) < abs(D_01):
            cost += abs(D_11)
            D_01 += D_11
            D_11 = 0
        else:
            cost += abs(D_01)
            D_11 += D_01
            D_01 = 0
    #print(cost, D_00, D_01, D_10, D_11)
    
    assert abs(D_00) == abs(D_11) and abs(D_01) == abs(D_01), "j'ai raté à l'aide"

    if D_00 != 0:
        cost += abs(D_00) * np.sqrt(2)
    if D_01 != 0:
        cost += abs(D_01) * np.sqrt(2)
    return cost/N

def Wasserstein_swap_bins(bins_1, bins_2):
    '''
    the bins are Y1S1, Y0S1, Y1S0, Y0S0
    '''
    sign = lambda x: math.copysign(1, x)
    
    D_11 = bins_1[0] - bins_2[0]
    D_10 = bins_1[1] - bins_2[1]
    D_01 = bins_1[2] - bins_2[2]
    D_00 = bins_1[3] - bins_2[3]
    N = bins_1.sum()
    cost = 0

    #Cost matrix is 0 in diag, sqrt(2) inv diag and 1 elsewhere, hence we do as much as possible on near points (1 cost instead of sqrt(2))
    if sign(D_00) != sign(D_01):
        if (abs(D_00) < abs(D_01)):
            cost += abs(D_00)
            D_01 += D_00
            D_00 = 0
        else:
            cost += abs(D_01)
            D_00 += D_01
            D_01 = 0
    #print(cost, D_00, D_01, D_10, D_11)

    if (sign(D_00) != sign(D_10)) and (D_00 != 0):
        if (abs(D_00) < abs(D_10)):
            cost += abs(D_00)
            D_10 += D_00
            D_00 = 0
        else:
            cost += abs(D_10)
            D_00 += D_10
            D_10 = 0
    #print(cost, D_00, D_01, D_10, D_11)

    if sign(D_11) != sign(D_10) and (D_10 != 0):
        if abs(D_11) < abs(D_10):
            cost += abs(D_11)
            D_10 += D_11
            D_11 = 0
        else:
            cost += abs(D_10)
            D_11 += D_10
            D_10 = 0
    #print(cost, D_00, D_01, D_10, D_11)

    if (sign(D_11) != sign(D_01)) and (D_01 != 0) and (D_11 != 0):
        if abs(D_11) < abs(D_01):
            cost += abs(D_11)
            D_01 += D_11
            D_11 = 0
        else:
            cost += abs(D_01)
            D_11 += D_01
            D_01 = 0
    #print(cost, D_00, D_01, D_10, D_11)
    
    assert abs(D_00) == abs(D_11) and abs(D_01) == abs(D_01), "j'ai raté à l'aide"

    if D_00 != 0:
        cost += abs(D_00) * np.sqrt(2)
    if D_01 != 0:
        cost += abs(D_01) * np.sqrt(2)
    return cost/N



###########################################
#------Hellinger distance-----------------#
###########################################

def Hellinger_distance(arr_1, arr_2):
    arr_1, arr_2 = transform_arr_dic_like_KL(arr_1, arr_2, intersection=False)
    return 1 - ((arr_1 * arr_2)**(1/2)).sum()

def Total_variation(arr_1, arr_2):
    arr_1, arr_2 = transform_arr_dic_like_KL(arr_1, arr_2, intersection=False)
    return (np.abs(arr_1 - arr_2)).sum()/2

def Jensen_shannon(arr_1, arr_2):
    arr_1, arr_2 = transform_arr_dic_like_KL(arr_1, arr_2, intersection=False)
    M = (arr_1 + arr_2)/2
    return (1/2)*(KL_fct(arr_1, M) + KL_fct(arr_2, M))