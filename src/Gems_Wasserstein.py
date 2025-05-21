import numpy as np    
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import torch
from GEMS3_base_explainer import *

class Stresser():
    def __init__(self,
                 X : np.array,
                 Y : np.array = None,
                 Pred : np.array = None,
                 S : np.array = None
                 ):
        self.X = X
        self.X_means = np.mean(X, axis = 0)
        self.Y = Y
        self.Pred = Pred
        self.S = S
        self.KL = obs_stresser(X)
        self.W  = Stresser_W(X, Y, Pred, S)
        self.n = len(X)

class Stresser_W():

    def __init__(self,
                 X : np.array,
                 Y : np.array = None,
                 Pred : np.array = None,
                 S : np.array = None,
                 ):
        self.X = X
        self.X_means = np.mean(X, axis = 0)
        self.Y = Y
        self.Pred = Pred
        self.t = np.zeros(X.shape)
        self.n = len(X)
        self.S = S

    def fit(self, dic, verbose=False):
        """
        Compute the optimal transport np.array vector to add to the original dataset given the condition on the dic
        Input parameters:
            - dic: a dictionnary which can have 5 different keys:
                - 'means'  : {id_column_1: value_1 , ... , id_column_n : value_n }
            - verbose: The accuracy of the result is shown if verbose==True
        """
        self.t = np.zeros(self.X.shape)
        if not 'means' in dic:
            dic['means'] = {}

        self.dic = dic

        #1) init
        for column_index, mean_value in dic['means'].items():
            if verbose:
                print(self.X_means.shape)
                print(column_index)
                print(self.X_means[column_index])
                print(mean_value - self.X_means[column_index])
            λ_star = mean_value - self.X_means[column_index]
            self.t[:,column_index] = λ_star

        return

    def fit_semi_discret(self, 
                         dic, 
                         verbose = False,
                         dist_precision : int = 4,
                         ):
        
        if not 'means' in dic:
            dic['means'] = {}

        self.t = np.zeros(self.X.shape)
        self.dic = dic
        sign_fct = lambda x: math.copysign(1, x)

        for column_index, mean_value in dic['means'].items():
            if verbose:
                print(f'We are constraining the column with {column_index} as index to {mean_value} as mean value')
            arr_actual = self.X[:,column_index].copy()
            assert mean_value < arr_actual.max() and mean_value > arr_actual.min(), 'the constraint applied cannot be applied, please put a mean value within the boundary of the column value'
            translation_column = np.zeros(len(arr_actual))
            translation_mean_needed = mean_value - self.X_means[column_index]
            
            #np.sort has no 'descending argument', .reverse creates another array -> optimal way
            #We will allow the value only to go to the right with this sorting
            t_mean_needed_sign = sign_fct(translation_mean_needed)
            unique_value_sorted = t_mean_needed_sign * np.sort(np.unique(arr_actual * t_mean_needed_sign))

            #The dist_arr and dic will allow us to navigate among the possible values without loosing time (and re-computing it each time)
            dist_arr = abs(unique_value_sorted[1:] - unique_value_sorted[:-1]).round(dist_precision)
            dic_value_index = {}
            for index, value in enumerate(unique_value_sorted):
                dic_value_index[value] = index
            
            #Function to find the distance to next point
            def fct_dist_next(value):
                index = dic_value_index[value]
                if index < len(dist_arr):
                    return dist_arr[index]
                return np.float32('inf')
            fct_dist_next_vect = np.vectorize(fct_dist_next)

            #Function to assign the next point (could also have been done using the sign & dist_to_next)
            fct_assign_next = lambda value : unique_value_sorted[dic_value_index[value]+1]
            fct_assign_next_vect = np.vectorize(fct_assign_next)

            #We continue adding the lest costly 
            diff_mean = abs(translation_mean_needed) - abs(translation_column.mean())

            #Distance to the next closest point in the direction of the objective (increasing or lowering the column mean)
            dist_to_next = fct_dist_next_vect(arr_actual)
            cost_to_next = dist_to_next**2

            #Interest : ratio distance / cost
            mod_interest = dist_to_next / cost_to_next

            while diff_mean > 0:
                # Because the last (extremum) point has a +inf dist to the next, there also is a +inf cost -> we ignore those elements with np.nanmax
                value = np.nanmax(mod_interest)
                indexs = np.where(mod_interest == value)[0]

                if diff_mean < (dist_to_next[indexs].sum() / self.n):
                    #Last iteration : we change just enought values to go over the constraint given
                    if verbose:
                        print('last iteration for this column')
                    
                    number_changement = int((diff_mean * self.n )//1) + 1
                    if verbose:
                        print(f'we modified {number_changement} of the {len(indexs)} points having the same next point impact on the Wasserstein distance')
                    indexs = indexs[:number_changement]
                    

                new_arr_indexs = fct_assign_next_vect(arr_actual[indexs])
                translation_column[indexs] += dist_to_next[indexs] * t_mean_needed_sign
                arr_actual[indexs] = new_arr_indexs

                #Update the distance to next point array
                dist_to_next[indexs] = fct_dist_next_vect(arr_actual[indexs])

                #Update the cost array : cost = (translation + dist_to_next)^2 - translation^2 (to pay minus what have already been paid)
                cost_to_next[indexs] = dist_to_next[indexs]**2 + 2 * dist_to_next[indexs] * abs(translation_column[indexs])

                #Uptade the modification interest
                mod_interest = dist_to_next / cost_to_next

                #Uptade the distance to contraint objectif
                diff_mean = abs(translation_mean_needed) - abs(translation_column.mean())
                if verbose:
                    print(f'the new mean difference is {diff_mean.round(3)}, we just added a translation of {translation_column[indexs][0]* t_mean_needed_sign} to obtain more {arr_actual[indexs][0]}')
            
            self.t[:,column_index] = translation_column

        return
  
    def grad_step_fct(self,
                      z_0,
                      z_t,
                      direction,
                      CONSTR_REGU = 1):
        inputs = z_t.detach().clone().float()
        inputs.requires_grad = True
        outputs = - self.network(inputs) * direction
        grad = torch.autograd.grad(outputs = outputs,
                                       inputs =  inputs,
                                       grad_outputs=torch.ones_like(outputs),
                                       create_graph=False,
                                       only_inputs=True)[0]
        grad_step = 2*(z_t - z_0) + CONSTR_REGU * grad
        return grad_step
    
    def fit_grad(self,
                 z_0,
                 prob_threshold,
                 threshold,
                 CONSTR_REGU,
                 threshold_augm_constr,
                 lr,
                 verbose,
                 iteration_threshold,
                 look_alike,
                 reachable,
                 **kwargs,
                 ):
        number_columns = self.X.shape[1]
        start_pred = self.network(z_0)
        start_pred_bin = ( start_pred > prob_threshold) * 1.
        direction = 2.* (start_pred_bin.mean() < threshold) - 1
        pred_bin = start_pred_bin
        stop_cdt = (threshold - start_pred_bin.mean()) * direction
        if verbose:
            print(f'the starting mean is {np.round(start_pred_bin.mean().item(), 3)}, we are {np.round(stop_cdt.item(), 3)} away')
            print(f'Thus the direction is {direction}')
            print()
        z_t = z_0
        iteration = 0
        former_stop_cdt = stop_cdt
        while stop_cdt > 0:
            if verbose:
                print(iteration)
            assert iteration < iteration_threshold, 'non convergence toward the objective, or too slow'
            
            #For the elements which we have already changed their prediction, we do not continue moving them
            index_to_consider = torch.from_numpy(np.where(pred_bin == start_pred_bin)[0])
                    
            z_t_new = z_t.clone()
            
            lr_for_loop = lr
            for i in range(6):
                grad_step = self.grad_step_fct(z_0 = z_0, 
                                                z_t = z_t_new,
                                                direction = direction,
                                                CONSTR_REGU = CONSTR_REGU,
                                                )
                
                z_t_new[index_to_consider] = z_t[index_to_consider] - lr * grad_step[index_to_consider]
                
                z_t_new_copy = z_t_new.clone()
                if look_alike:
                    for i in range(number_columns):
                        indexs = torch.argmin(abs(reachable[i].reshape((1,-1)).repeat_interleave(len(z_t_new), axis = 0) - z_t_new[:,i].reshape((-1, 1)).repeat_interleave(len(reachable[i]), axis = 1)),
                                            axis = 1)
                        z_t_new[:,i] = reachable[i][indexs]

                pred = self.network(z_t_new)
                lr_for_loop = lr_for_loop / 1.2
                pred_bin = (pred > prob_threshold) * 1.
                stop_cdt = (threshold - pred_bin.mean() ) * direction
                
                if stop_cdt < 0:
                    if verbose:
                        print(stop_cdt)
                    return z_t_new.detach().numpy(), pred_bin.mean().item()
                
            z_t = z_t_new_copy
            if (former_stop_cdt - stop_cdt) < threshold_augm_constr:
                CONSTR_REGU = CONSTR_REGU * 1.2
                z_t = z_0
                if verbose:
                    print(f'the former constraint was equal to {np.round(former_stop_cdt.item(), 5)}, the new is {np.round(stop_cdt.item(), 5)}')
                    print(f'the regularization constraint increased to {np.round(CONSTR_REGU, 5)}')
                    #print(f'the learning rate increased to {np.round(lr, 5)}')
            if verbose:
                print(f'the contraint value is {np.round(stop_cdt.item(), 5)}')
                print(f'the euclidian distance between is {((z_t - z_0)**2).sum(axis = 1).mean()}, \n')
            former_stop_cdt = stop_cdt
            iteration +=1

        return z_t_new.detach().numpy(), pred_bin.mean().item()
      
    def DI_miti_grad(self,
                     network,
                     #typ = 'proportional',
                     prob_threshold = 0.5,
                     threshold = 0.8,
                     CONSTR_REGU = 0.1,
                     threshold_augm_constr = 0.01,
                     threshold_lr_cdt = 0.001,
                     lr = 0.001,
                     iteration_threshold = 100,
                     look_alike = False,
                     verbose = False,
                     delta_type = 'mean'
                     ):
        '''
        This function creates the closest for W2 X counterfactual while mitigating DI bias
        optimization prblm : min |Z - Z_0|^2 <=> W_2(Z,Z_0) s.t. DI(f(Z, S)) > threshold
        opt rewritten : min W_2(Z,Z_0) + CONSTR_REGU * direction * ( (threshold +/- delta - f(Z) ) for each S=0 or S=1
        
        INPUT:
        network : torch neural network which we will use for its gradient
        threshold : define the DI constraint -> between 0 and 1 non included
        prob_threshold : define the logits threshold : the logits after a sigmoid are within [0,1] and we can instead of choosing the regular 0.5 choose another threshold
        CONSTR_REGU : it defined how much importance we place on the constraint: we want this variable to be as low as possible while respecting the constraint.
        threshold_augm_constr : threshold which will define when to increase the CONSTR_REGU (identify convergence of the gradient step)
        lr : learning rate of the gradient descent
        iteration_threshold : number of max iteration -> if after this number of iteration the constraint still is not reached, 
                              we suggest to either increasing the CONSTR_REGU, this threshold, or to verify that the the network's output comes directly from the inputs (grad != 0) 
        look_alike : whether the achievable value are only from the 1D achieved values : 1D-projection
        verbose : whether you prefer to have print showing you what is happening in real time
        
        OUTPUT:
        None, the result is stored in the self.t which is the translation to do to the network input (to self.X) to unbias the original dataset and have a DI > threshold
        
        '''
        self.t = np.zeros(self.X.shape)
        network.eval()
        self.network = network
        n1, n0 = self.S.sum(), (1-self.S).sum()
        index_S0, index_S1 = np.where(self.S == 0), np.where(self.S == 1)
        pred = (((network(torch.from_numpy(self.X).type(torch.float32))) >prob_threshold)*1.).squeeze()
        P_S0, P_S1 = (pred[index_S0].mean()).item(), (pred[index_S1].mean()).item()
        DI, lambda_0, lambda_1 = P_S0 / P_S1, P_S0 * n0, P_S1 * n1
        assert DI < threshold, 'nothing to do, the DI is already above the threshold'
        Diff_DI = threshold - DI
        if verbose:
            print(f'P(Y=1|S=0) = {np.round(P_S0, 4)}, P(Y=1|S=1) = {np.round(P_S1, 4)}')
            print(f'former DI is {np.round(DI, 3)}, we thus have a difference of {np.round(Diff_DI, 3)} to mitigate')

        if delta_type == 'mean': 
                #Same mean : proportional case
                delta_1_mean = lambda_1 / (1 + (1 + (n1 * lambda_0)/(n0 * lambda_1)) / Diff_DI) / n1
                delta_0_mean = delta_1_mean
        else:
            #same number of changement : balanced case
            delta_1 = lambda_1 / (1 + ( n1 / ( n0 * Diff_DI) ) * (1 + lambda_0 / lambda_1 ) ) 
            delta_1_mean = delta_1 / n1
            delta_0_mean = delta_1 / n0

        threshold_1 = P_S1 - delta_1_mean
        threshold_0 = P_S0 + delta_0_mean

        if verbose:
            print(f'We want to change it to new_P(Y=1|S=0) = {np.round(threshold_0, 4)}, and new_P(Y=1|S=1) = {np.round(threshold_1, 4)}')
        index_00, index_11 = np.where((self.S == 0) & (pred.squeeze().numpy() == 0))[0], np.where((self.S == 1) & (pred.squeeze().numpy() == 1))[0]
        z_0_0, z_0_1 = self.X[index_00], self.X[index_11]
        
        n_threshold_0, n_threshold_1 = (threshold_0 - P_S0) * n0 / (n0 - lambda_0), threshold_1 * n1 / lambda_1
        
        if verbose:
            print(n_threshold_0, n_threshold_1)
            
        #We keep in memory the achieved value for each columns
        number_column = self.X.shape[1]
        reachable = [0]*number_column
        for i in range(number_column):
          reachable[i] = torch.from_numpy(np.unique(self.X[:,i])).float()
        
        z_t_0, new_P_S0 = self.fit_grad(torch.from_numpy(z_0_0).float(),
                                        prob_threshold = prob_threshold,
                                        threshold = n_threshold_0,
                                        CONSTR_REGU = CONSTR_REGU,
                                        threshold_augm_constr = threshold_augm_constr,
                                        threshold_lr_cdt = threshold_lr_cdt,
                                        lr = lr,
                                        stop_condition_threshold = stop_condition_threshold,
                                        verbose = verbose,
                                        iteration_threshold = iteration_threshold,
                                        look_alike = look_alike,
                                        reachable = reachable
                                        )
        self.z_t_0 = z_t_0
        
        new_pred_z_t_0   = (self.network(torch.from_numpy(z_t_0).float()) > prob_threshold)*1.
        pred_mod_and_min = np.where(new_pred_z_t_0 != 0)[0][:(math.ceil(n_threshold_0 * len(index_00)))]
        t_0 = np.zeros(z_t_0.shape)
        t_0[pred_mod_and_min] = z_t_0[pred_mod_and_min] - z_0_0[pred_mod_and_min]  

        if verbose:
            print(new_P_S0, n_threshold_0, ((t_0 != 0).sum(axis = 1) > 0).mean())
            
        z_t_1, new_P_S1 = self.fit_grad(torch.from_numpy(z_0_1).float(),
                                        prob_threshold = prob_threshold,
                                        threshold = n_threshold_1,
                                        CONSTR_REGU = CONSTR_REGU,
                                        threshold_augm_constr = threshold_augm_constr,
                                        threshold_lr_cdt = threshold_lr_cdt,
                                        lr = lr,
                                        stop_condition_threshold = stop_condition_threshold,
                                        verbose = verbose,
                                        iteration_threshold = iteration_threshold,
                                        look_alike = look_alike,
                                        reachable = reachable,
                                        )
        self.z_t_1 = z_t_1
        

        new_pred_z_t_1   = (self.network(torch.from_numpy(z_t_1).float()) > prob_threshold)*1.
        pred_mod_and_min = np.where(new_pred_z_t_1 != 1)[0][:(math.ceil((1 - n_threshold_1) * len(index_11)))]
        t_1 = np.zeros(z_t_1.shape)
        t_1[pred_mod_and_min] = z_t_1[pred_mod_and_min] - z_0_1[pred_mod_and_min]  
        if verbose:
            print(new_P_S1, n_threshold_1, ((t_1 != 0).sum(axis = 1) > 0).mean())
    
        self.t[index_00,:] = t_0
        self.t[index_11,:] = t_1

        return

    
    def print_stats(self,col_id):
        """
        print elementary statistics on the column 'col_id'
        """
        print("min:",  np.min(self.X[:,col_id]))
        print("max:",  np.max(self.X[:,col_id]))
        print("mean:", np.mean(self.X[:,col_id]))
        print("std:",  np.std(self.X[:,col_id]))

    def get_lambda(self):
        return self.t
    
    def get_modified_array(self, 
                           dic = None,
                           dist_precision = 4,
                           verbose = False,
                           stress_type = 'semi_discret',
                           ):
        
        if not(dic is None):
            if stress_type == 'semi_discret':
                self.fit_semi_discret(dic = dic, 
                                      verbose=verbose,
                                      dist_precision = dist_precision)
            else:
                self.fit(dic = dic, verbose=verbose)

        return self.X + self.t
    
    def return_distance(self, dist_fct):
        return dist_fct(self.X, self.get_modified_array())
    
    def show_stress_variable(self, 
                             list_column_index, 
                             dic_index_to_name,
                             obs_fct = lambda x,y,pred : pred.mean(),
                             same_plot = False,
                             return_df = False,
                             verbose = False,
                             y_name = 'result_fct',
                             ):

        df_result = pd.DataFrame({})

        for column_index in list_column_index:
            list_result = []
            arr = self.X[:,column_index]
            if len(np.unique(arr)) > 10:
                list_stress_level = np.quantile(a = arr, q = np.arange(0.25, 0.751, 0.05))
            else:
                #We do not take the regular quantile approach with binary or numerical variable with fewer than 10 values
                ma, mi, me = arr.max(), arr.min(), arr.mean()
                inter_max, inter_min = (me + mi)/2, (ma + me)/2

                # we take the points between the extermities and the mean, divided by 11 because there are 11 quantiles from 0.25 to 0.75 included
                list_stress_level = np.arange(start = inter_min, 
                                              stop = inter_max, 
                                              step = (inter_max - inter_min) / 11)
                if verbose:
                    print(list_stress_level)
                
            
            for stress_value in list_stress_level :
                mod_arr = self.get_modified_array(dic = {'means' : {column_index : stress_value}})
                result = obs_fct(mod_arr[:,column_index], 
                                 self.Y, 
                                 self.Pred,
                                 self.X[:,column_index],
                                 self.t[:,column_index],
                                 stress_value - arr.mean()
                                 )
                list_result.append(result)

            if verbose:
                print('list_stress')
                print(list_stress_level)
                print('list result')
                print(list_result)
                print('column name')
                print( dic_index_to_name[column_index])

            df_result = pd.concat([df_result, 
                                   pd.DataFrame({'stress_value' : list(list_stress_level) + [arr.mean()], 
                                                y_name : list(list_result) + [0],
                                                'column_name'  : dic_index_to_name[column_index]
                                                }),
                                    ])

        if verbose:
            print(df_result.columns)
        
        #Below we plot the results
        if same_plot:
            sns.lineplot(data = df_result, 
                         x = 'stress_value',
                         y = y_name,
                         hue = 'column_name')
        else:
            fig, ax = plt.subplots(1, 
                                   len(list_column_index), 
                                   figsize = (len(list_column_index)*5, 5)  
                                   )
            for index, column_name in enumerate(df_result.column_name.unique()):
                sns.lineplot(data = df_result[df_result.column_name == column_name], 
                             x = 'stress_value',
                             y = y_name,
                             ax = ax[index])
                ax[index].set_title(column_name)
        
        plt.show()
        if return_df:
            return df_result
        return





