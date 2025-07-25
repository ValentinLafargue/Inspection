import numpy as np
import pandas as pd
from utils import *
import sys 
sys.path.append("/content/drive/MyDrive/Inspection/Project")
from GEMS3_base_explainer import *
import ot

def Gems_regular_mitigation(df, 
                            S_name, 
                            Pred_name, 
                            DI_target, 
                            dic_name_to_index_mapping,
                            gd_iterations = 1000,
                            *args,
                            **kwargs):
    df_mod =  df.copy()
    Pred_column_index = dic_name_to_index_mapping[Pred_name]
    S_column_index = dic_name_to_index_mapping[S_name]

    Gems = obs_stresser(df_mod.values)
    Gems.fit({'DI' : [Pred_column_index, 
                      S_column_index, 
                      DI_target]},
            gd_iterations = gd_iterations,
            )
    lambdas = Gems.get_lambda()
    df_mod['lambda'] = lambdas
    return df_mod
    
def Gems_fair_mitigation(df, 
                         S_name, 
                         Y_name, 
                         Pred_name, 
                         DI_target, 
                         dic_name_to_index_mapping,
                         gd_iterations = 1000,
                         delta_type = 'mean',
                         *args,
                         **kwargs):
    df_mod =  df.copy()
    Pred_column_index = dic_name_to_index_mapping[Pred_name]
    S_column_index = dic_name_to_index_mapping[S_name]
    Y_column_index = dic_name_to_index_mapping[Y_name]

    Gems_fair = obs_stresser_fairness(df_mod.values, 
                                      Id_col_Ypred = Pred_column_index, 
                                      Id_col_S = S_column_index, 
                                      fairness_type='DI', 
                                      Id_col_Ytrue= Y_column_index)
    Gems_fair.fit_and_cpt_lambdas(fairness_value = DI_target, 
                                  verbose = False,
                                  gd_iterations = gd_iterations,
                                  delta_type = delta_type)
    lambdas = Gems_fair.get_lambda()
    df_mod['lambda'] = lambdas
    return df_mod

def Gems_regular_mitigation_arr(arr, 
                            S_column_index,
                            Pred_column_index, 
                            DI_target = 0.8,
                            gd_iterations = 1000,
                            **kwargs,):
    Gems = obs_stresser(arr)
    Gems.fit({'DI' : [Pred_column_index, 
                      S_column_index, 
                      DI_target]},
             gd_iterations = gd_iterations)
    lambdas = Gems.get_lambda()
    arr_mod = np.concatenate([arr, lambdas.reshape(len(lambdas), -1)], axis = 1)
    return arr_mod
    
def Gems_fair_mitigation_arr(arr, 
                            S_column_index,
                            Pred_column_index, 
                            Y_column_index = None,
                            DI_target = 0.8,
                            gd_iterations = 1000,
                            delta_type = 'mean',
                            **kwargs,):
    Gems_fair = obs_stresser_fairness(arr, 
                                      Id_col_Ypred = Pred_column_index, 
                                      Id_col_S = S_column_index, 
                                      fairness_type='DI', 
                                      Id_col_Ytrue= Y_column_index)
    Gems_fair.fit_and_cpt_lambdas(fairness_value = DI_target,
                                  verbose = False,
                                  gd_iterations = gd_iterations,
                                  delta_type = delta_type)

    lambdas = Gems_fair.get_lambda()
    arr_mod = np.concatenate([arr, lambdas.reshape(len(lambdas), -1)], axis = 1)
    return arr_mod

def discr_alg_empowering(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     **kwargs):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    It modifies the algorithm's decision for the minority group (generally S = 0)
    '''
    
    df_mod = df.copy()
    index = np.where((df_mod[S_name] == 0) & (df_mod[Pred_name] == 0))[0]
    i = 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        df_mod.loc[index[i],Pred_name] = 1
        i += 1
        #print(DI(df_mod, Pred_name))
    return df_mod

def discr_alg_belittling(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     **kwargs):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    It modifies the algorithm's decision for the majority group (S = 1)
    '''
    df_mod = df.copy()
    index = np.where((df_mod[S_name] == 1) & (df_mod[Pred_name] == 1))[0]
    i = 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        df_mod.loc[index[i],Pred_name] = 0
        i += 1
        #print(DI(df_mod, Pred_name))
    return df_mod

def discr_alg_mixed(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     verbose : bool = False ,
                     **kwargs,
                     ):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    It alternates between modifying the algorithm's decision for the minority group and for the majority group
    '''
    df_mod = df.copy()
    index_1 = np.where((df_mod[S_name] == 0) & (df_mod[Pred_name] == 0))[0]
    index_2 = np.where((df_mod[S_name] == 1) & (df_mod[Pred_name] == 1))[0]
    i, i_1, i_2 = 0, 0, 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        if verbose:
            print(DI_fct_dataset(df_mod, Pred_name))
        
        if i % 2 == 0:
            df_mod.loc[index_1[i_1],Pred_name] = 1
            i_1 += 1
        else:
            df_mod.loc[index_2[i_2],Pred_name] = 0
            i_2 += 1
        i += 1
    return df_mod

def discr_att_empowering(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     **kwargs):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    It modifies the individual group when the algorithm decision is positive (Pred_name = 1)
    '''
    df_mod = df.copy()
    index = np.where((df_mod[S_name] == 1) & (df_mod[Pred_name] == 1))[0]
    i = 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        df_mod.loc[index[i],S_name] = 0
        i += 1
        #print(DI(df_mod))
    return df_mod

def discr_att_belittling(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     **kwargs):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    It modifies the individual group when the algorithm decision is negative (Pred_name = 0)
    '''
    df_mod = df.copy()
    index = np.where((df_mod[S_name] == 0) & (df_mod[Pred_name] == 0))[0]
    i = 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        df_mod.loc[index[i],S_name] = 1
        i += 1
        #print(DI(df_mod))
    return df_mod

def discr_att_mixed(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     **kwargs):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    It alternates between modifying the individual group when the algorithm decision is positive or is negative.
    '''
    df_mod = df.copy()
    index_1 = np.where((df_mod[S_name] == 1) & (df_mod[Pred_name] == 1))[0]
    index_2 = np.where((df_mod[S_name] == 0) & (df_mod[Pred_name] == 0))[0]
    i, i_1, i_2 = 0, 0, 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        if i % 2 == 0:
            df_mod.loc[index_1[i_1],S_name] = 0
            i_1 += 1
        else:
            df_mod.loc[index_2[i_2],S_name] = 1
            i_2 += 1
        i += 1
        #print(DI(df_mod))
    return df_mod

def discr_all_mixed(df : pd.DataFrame, 
                     S_name : str, 
                     Pred_name : str, 
                     DI_target : float,
                     *args,
                     verbose :bool = False,
                     **kwargs):
    '''
    This functions takes a dataframe and returns a modified dataframe with the same shape but with a DI of Pred_name higher than DI_target
    '''
    df_mod = df.copy()
    index_1 = np.where((df_mod[S_name] == 1) & (df_mod[Pred_name] == 1))[0]
    index_2 = np.where((df_mod[S_name] == 0) & (df_mod[Pred_name] == 0))[0]
    i, i_1, i_2= 0, 0, 0
    while DI_fct_dataset(df_mod, Pred_name) < DI_target:
        if i_1 < len(index_1):
            if i % 4 == 0:
                df_mod.loc[index_1[i_1],S_name] = 0
                i_1 += 1
            if i % 4 == 2:
                df_mod.loc[index_1[i_1],Pred_name] = 1
                i_1 += 1
        if i_2 < len(index_2):
            if i % 4 == 1:
                df_mod.loc[index_2[i_2],S_name] = 1
                i_2 += 1
            if i % 4 == 3:
                df_mod.loc[index_2[i_2],Pred_name] = 0
                i_2 += 1
        i += 1
        if verbose:
            print(DI_fct_dataset(df_mod, Pred_name))
    return df_mod


def DI_mitigation_bins_wasserstein(bins, 
                              threshold = 0.8, 
                              speed = 1,
                              modifying_S = True,
                              verbose = False):
    '''
    Here we seek to mitigate the DI bias by modifying the F(x) and S if necessaries conditions
    '''
    #Y0S0 -> Y1S0, Y0S0 -> Y0S1, Y1S1 -> Y1S0
    n1, n0 = bins[0] + bins[1], bins[2] + bins[3]
    PY1S1, PY1S0 = bins[0]/n1, bins[2] / n0
    DI =  PY1S0 / PY1S1

    #We need to create the list because dic.keys gives a set in a pseudo-random order
    swap_possible = ['Y0S0 -> Y1S0', 'Y0S0 -> Y0S1', 'Y1S1 -> Y1S0']
    dic_swap_done = {'Y0S0 -> Y1S0' : 0, 
                     'Y0S0 -> Y0S1' : 0, 
                     'Y1S1 -> Y1S0' : 0
    }
    DI_n = np.zeros(3)
    arr_bins = np.zeros((3,4))
    while DI < threshold:
        bins_n = bins + np.array([0, 0, 1, -1])
        arr_bins[0,:] = bins_n.copy()
        n1_n, n0_n = bins_n[0] + bins_n[1], bins_n[2] + bins_n[3]
        PY1S1_n, PY1S0_n = bins_n[0]/n1_n, bins_n[2] / n0_n
        DI_n[0] =  PY1S0_n / PY1S1_n

        if modifying_S:
            bins_n = bins + np.array([0, 1, 0, -1])
            arr_bins[1,:] = bins_n.copy()
            n1_n, n0_n  = bins_n[0] + bins_n[1], bins_n[2] + bins_n[3]
            PY1S1_n, PY1S0_n = bins_n[0]/n1_n, bins_n[2] / n0_n
            DI_n[1] =  PY1S0_n / PY1S1_n

            bins_n = bins + np.array([-1, 0, 1, 0])
            arr_bins[2,:] = bins_n.copy()
            n1_n, n0_n = bins_n[0] + bins_n[1], bins_n[2] + bins_n[3]
            PY1S1_n, PY1S0_n = bins_n[0]/n1_n, bins_n[2] / n0_n
            DI_n[2] =  PY1S0_n / PY1S1_n

        j = np.argmax(DI_n)
        dic_swap_done[swap_possible[j]] += speed
        bins = bins + speed * (arr_bins[j, :] - bins)
        n1, n0 = bins[0] + bins[1], bins[2] + bins[3]
        PY1S1, PY1S0 = bins[0]/n1, bins[2] / n0
        DI =  PY1S0 / PY1S1
        if verbose:
            print(DI)
    return bins, dic_swap_done
    
def DI_mitigation_inds_wasserstein(
                              arr,
                              S_index,
                              Y_index,
                              threshold = 0.8,
                              verbose = False,
                              metric_fct_bins = calculate_DI_bins,
                              sign_threshold = 1,
                              ):
    '''
    Here we seek to mitigate the DI bias by using a different sampling
    similar to DI_mitigation_bins_wasserstein but we take into account the wasserstein cost 
    for the transport between the different X (we do not only modify f(X)).
    For that reason, we also add Y1S1 -> Y0S1
    '''
    
    S1_index = arr[:,S_index] == 1
    Y1_index = arr[:,Y_index] == 1
    
    # ~ reverse the index
    Y1S1_index, Y0S1_index, Y1S0_index, Y0S0_index =  S1_index & Y1_index ,S1_index & (~Y1_index),(~S1_index) & Y1_index ,(~S1_index) & (~Y1_index)
    
    #Next line just for clarity
    Y1S1, Y0S1, Y1S0, Y0S0 = len(arr[Y1S1_index]), len(arr[Y0S1_index]), len(arr[Y1S0_index]), len(arr[Y0S0_index])
    bins = np.array([Y1S1, Y0S1, Y1S0, Y0S0], dtype = int)
    
    DI =  metric_fct_bins(bins)
    
    #We need to create the list because dic.keys gives a set in a pseudo-random order
    swap_possible = ['Y0S0 -> Y1S0', 'Y0S0 -> Y0S1', 'Y1S1 -> Y1S0', 'Y1S1 -> Y0S1']
    dic_swap_impact = {'Y0S0 -> Y1S0' : [0,  0, 1, -1], 
                       'Y0S0 -> Y0S1' : [0,  1, 0, -1], 
                       'Y1S1 -> Y1S0' : [-1, 0, 1,  0],
                       'Y1S1 -> Y0S1' : [-1, 1, 0,  0],
    }
    dic_number_swap_done = {'Y0S0 -> Y1S0' : 0, 
                            'Y0S0 -> Y0S1' : 0, 
                            'Y1S1 -> Y1S0' : 0,
                            'Y1S1 -> Y0S1' : 0,
    }
    dic_swap_index_to_index = {'Y0S0 -> Y1S0' : [Y0S0_index, Y1S0_index], 
                               'Y0S0 -> Y0S1' : [Y0S0_index, Y0S1_index],
                               'Y1S1 -> Y1S0' : [Y1S1_index, Y1S0_index],
                               'Y1S1 -> Y0S1' : [Y1S1_index, Y0S1_index],
    }
    swaps = []
    M = ot.dist(x1 = arr, x2 = arr, p=2, metric = 'euclidean')
    dic_swap_cost = {}
    for swap in swap_possible:
        dic_swap_cost[swap] = np.zeros((dic_swap_index_to_index[swap][0].sum(), 3))
        
    for swap in swap_possible:
        M_copy = M.copy()
        M_copy[~dic_swap_index_to_index[swap][0],:] = M.max()+1
        M_copy[:,~dic_swap_index_to_index[swap][1]]  = M.max()+1 
        argmins = np.argmin(M_copy, axis = 1)
        for count, index in enumerate(np.where(dic_swap_index_to_index[swap][0])[0]):
            index_nearest_swap_point = argmins[index]
            dic_swap_cost[swap][count,:] = [index, index_nearest_swap_point, M[index, index_nearest_swap_point]]
        
    DI_n          = np.zeros(4)
    lowest_cost_n = np.zeros(4) - 1e-6
    index_from_n, index_to_n = np.zeros(4), np.zeros(4)
    arr_bins      = np.zeros((4,4))
    while (sign_threshold * DI) < (sign_threshold * threshold):
        if verbose:
            print(DI)
        for count, swap in enumerate(swap_possible):
            index_lowest_cost = np.argmin(dic_swap_cost[swap][:,2])
            index_from_n[count], index_to_n[count], lowest_cost_n[count] = dic_swap_cost[swap][index_lowest_cost,:]
    
            arr_bins[count,:] = bins.copy() + dic_swap_impact[swap]
            DI_n[count] = metric_fct_bins(arr_bins[count,:])
    
    
        mod_DI_n = DI_n - DI
        best_swap_values_n = mod_DI_n / lowest_cost_n
        j = np.argmax(best_swap_values_n)
        index_from, index_to = index_from_n[j], index_to_n[j]
        for swap in swap_possible:
          ind_possible = np.where(dic_swap_cost[swap][:,0] == index_from)[0]
          dic_swap_cost[swap][ind_possible,2] = M.max()
        
        
        swaps.append([index_from, index_to, lowest_cost_n[j]])
       
    
        
        dic_number_swap_done[swap_possible[j]] += 1
        bins, DI = arr_bins[j, :].copy(), DI_n[j]
    swaps = np.array(swaps)
    return swaps, dic_number_swap_done

def find_sampling_wasserstein_DI(arr,
                                 S_index,
                                 Y_index,
                                 threshold = 0.8,
                                 verbose = False,
                                 metric_fct_bins = calculate_DI_bins,
                                 sign_threshold = 1,
                                 **kwargs,
                                ):
                                    
    swaps, dic_number_swap_done = DI_mitigation_inds_wasserstein(arr = arr.copy(),
                                                                 S_index = S_index,
                                                                 Y_index = Y_index,
                                                                 threshold = threshold,
                                                                 verbose = verbose,
                                                                 metric_fct_bins = metric_fct_bins,
                                                                 sign_threshold = sign_threshold,
                                                                 )
                                                                 
    wass = 0
    new_arr = arr.copy()
    for index_from, index_to, cost in swaps:
      new_arr[int(index_from),:] = arr[int(index_to),:]
      wass += cost
    
    return new_arr, swaps, dic_number_swap_done, wass/len(arr)
                                  
                                  
                                  
def find_translation_DI(arr,
                        S_index,
                        Y_index,
                        threshold,
                        speed,
                        bool_return_all,
                        verbose = False,
                        **kwargs):
    #Calculate the bins
    Se1_index = arr[:,S_index] == 1
    Ye1_index = arr[:,Y_index] == 1

    # ~ reverse the index
    Y1S1_index, Y0S1_index, Y1S0_index, Y0S0_index =  Se1_index & Ye1_index ,Se1_index & (~Ye1_index),(~Se1_index) & Ye1_index ,(~Se1_index) & (~Ye1_index)

    #Next line just for clarity
    Y1S1, Y0S1, Y1S0, Y0S0 = len(arr[Y1S1_index]), len(arr[Y0S1_index]), len(arr[Y1S0_index]), len(arr[Y0S0_index])
    bins = np.array([Y1S1, Y0S1, Y1S0, Y0S0], dtype = int)
    if verbose:
        print('former bins', bins)

    #Calculate the translation
    new_bins, dic_swap_done = DI_mitigation_bins_wasserstein(bins, threshold=threshold, speed=speed, verbose= verbose)
    if verbose:
        print('new bins', new_bins)
        print('translation dic', dic_swap_done)
    translation = np.zeros(arr.shape)
    #['Y0S0 -> Y1S0', 'Y0S0 -> Y0S1', 'Y1S1 -> Y1S0']

    #Creating the random translation array
    Y0S0_value_index, Y1S1_value_index = np.where(Y0S0_index ==1)[0], np.where(Y1S1_index == 1)[0]
    Y0S0_sub_indexing = np.random.choice(len(Y0S0_value_index), 
                             size = dic_swap_done['Y0S0 -> Y1S0'] + dic_swap_done['Y0S0 -> Y0S1'],
                             replace = False)
    Y1S1_sub_indexing = np.random.choice(len(Y1S1_value_index), 
                            size = dic_swap_done['Y1S1 -> Y1S0'],
                            replace = False)
    Y0S0_to_swap = Y0S0_value_index[Y0S0_sub_indexing]
    Y1S1_to_swap = Y1S1_value_index[Y1S1_sub_indexing]
    if dic_swap_done['Y0S0 -> Y1S0']>0:
        Y0S0_to_swap_Y = Y0S0_to_swap[:dic_swap_done['Y0S0 -> Y1S0']]
        if verbose:
            print(len(translation[Y0S0_to_swap_Y, Y_index]))
        translation[Y0S0_to_swap_Y, Y_index] = 1
    if dic_swap_done['Y0S0 -> Y0S1']>0:
        Y0S0_to_swap_S = Y0S0_to_swap[dic_swap_done['Y0S0 -> Y1S0']:]
        if verbose:
            print(len(translation[Y0S0_to_swap_S, S_index]))
        translation[Y0S0_to_swap_S, S_index] = 1
    if dic_swap_done['Y1S1 -> Y1S0']>0:
        if verbose:
            print(len(translation[Y1S1_to_swap,S_index]))
        translation[Y1S1_to_swap,S_index]   = -1
    
    new_arr = arr + translation
    if bool_return_all:
        return new_arr, translation, bins
    return new_arr
