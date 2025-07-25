import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import torch
import torch.nn as nn


import math
#from pandas.api.types import is_numeric_dtype

class Network_old(nn.Module):
    def __init__(self, init_column, 
                 activation_bool = False,
                 last_batch_norm = True,
                 n_nodes = 512,
                 n_loop = 4,
                 ):
        super().__init__()
        self.activation_bool = activation_bool
        self.seq = nn.Sequential()
        self.seq.append(nn.BatchNorm1d(init_column))
        self.seq.append(nn.Linear(init_column, n_nodes))
        self.seq.append(nn.ReLU())

        for i in range(n_loop):
            self.seq.append(nn.BatchNorm1d(n_nodes))
            self.seq.append(nn.Linear(n_nodes, n_nodes))
            self.seq.append(nn.ReLU())
        
        if last_batch_norm:
            self.seq.append(nn.BatchNorm1d(n_nodes))
        self.seq.append(nn.Linear(n_nodes, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        prob = self.seq(x)
        if self.activation_bool:
            return self.activation(prob)
        return prob
    

class Network(nn.Module):
    def __init__(self, init_column, 
                 activation_bool = False,
                 last_batch_norm = True,
                 n_nodes = 512,
                 n_loop = 4,
                 ):
        super().__init__()
        self.activation_bool = activation_bool
        self.seq = nn.Sequential()
        self.seq.append(nn.BatchNorm1d(init_column))
        self.seq.append(nn.Linear(init_column, n_nodes))
        self.seq.append(nn.ReLU())

        for i in range(n_loop):
            self.seq.append(nn.BatchNorm1d(n_nodes))
            self.seq.append(nn.Linear(n_nodes, n_nodes))
            self.seq.append(nn.ReLU())
        
        if last_batch_norm:
            self.seq.append(nn.BatchNorm1d(n_nodes))
        self.seq.append(nn.Linear(n_nodes, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        prob = self.seq(x)
        if self.activation_bool:
            return self.activation(prob)
        return prob
    
class Network_dropout(nn.Module):
    def __init__(self, init_column, 
                 activation_bool = False,
                 n_nodes = 512,
                 n_loop = 4,
                 ):
        super().__init__()
        self.activation_bool = activation_bool
        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(init_column, n_nodes))
        self.seq.append(nn.ReLU())

        for i in range(n_loop):
            self.seq.append(nn.Dropout(0.2))
            self.seq.append(nn.Linear(n_nodes, n_nodes))
            self.seq.append(nn.ReLU())

        self.seq.append(nn.Linear(n_nodes, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        prob = self.seq(x)
        if self.activation_bool:
            return self.activation(prob)
        return prob
    

def training_network_threshold(model,
                     optimizer,
                     threshold,
                     X_train,
                     Y_train,
                     X_test,
                     Y_test,
                     epochs,
                     batch_size,
                     ):
    list_loss_train, list_acc_train = np.zeros((epochs, math.ceil(len(X_train)//batch_size))), np.zeros((epochs, math.ceil(len(X_train)//batch_size)))
    list_loss_test,  list_acc_test  = np.zeros((epochs, math.ceil(len(X_test)//batch_size))),  np.zeros((epochs, math.ceil(len(X_test)//batch_size)))
    for epoch in range(1,epochs+1):
        #Train
        model.train()
        optimizer.train()


        for batch_count in range(math.ceil(len(X_train)//batch_size)):
            optimizer.zero_grad()
            x,y = X_train[(batch_count*batch_size):((batch_count+1)*batch_size)].float(), (Y_train[(batch_count*batch_size):((batch_count+1)*batch_size)] > threshold).float()
            output = model(x)
            batch_loss = ((y - output.squeeze())**2).mean() #loss(y, output.squeeze())
            batch_loss.backward()
            optimizer.step()

            acc = (1.*(y == (1.*(output.squeeze() > 0.5)))).mean()

            list_loss_train[epoch-1, batch_count] = batch_loss.item()
            list_acc_train[epoch-1, batch_count]  =  acc.item()

        with torch.no_grad():
            optimizer.eval()
            for batch_count in range(5):
                x = X_train[(batch_count*batch_size):((batch_count+1)*batch_size)].float()
                output = model(x)
        model.eval()

        #Test
        for batch_count in range(math.ceil(len(X_test)//batch_size)):
            x,y = X_test[(batch_count*batch_size):((batch_count+1)*batch_size)].float(), (Y_test[(batch_count*batch_size):((batch_count+1)*batch_size)] > threshold).float()
            output = model(x.float())
            batch_loss =  ((y - output.squeeze())**2).mean() #loss(y, output.squeeze())
            acc = (1.*(y == (1.*(output.squeeze() > 0.5)))).mean()

            list_loss_test[epoch-1, batch_count] = batch_loss.item()
            list_acc_test[epoch-1, batch_count]  = acc.item()
    return

def training_network_regression(model,
                                optimizer,
                                X_train,
                                Y_train,
                                X_test,
                                Y_test,
                                epochs,
                                batch_size,
                                ):
    
    for epoch in range(1,epochs+1):
        #Train
        model.train()
        optimizer.train()


        for batch_count in range(math.ceil(len(X_train)//batch_size)):
            optimizer.zero_grad()
            x,y = X_train[(batch_count*batch_size):((batch_count+1)*batch_size)].float(), (Y_train[(batch_count*batch_size):((batch_count+1)*batch_size)]).float()
            output = model(x)
            batch_loss = ((y - output.squeeze())**2).mean() #loss(y, output.squeeze())
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            optimizer.eval()
            for batch_count in range(5):
                x = X_train[(batch_count*batch_size):((batch_count+1)*batch_size)].float()
                output = model(x)
        model.eval()

        #Test
        for batch_count in range(math.ceil(len(X_test)//batch_size)):
            x,y = X_test[(batch_count*batch_size):((batch_count+1)*batch_size)].float(), (Y_test[(batch_count*batch_size):((batch_count+1)*batch_size)]).float()
            output = model(x.float())
            batch_loss =  ((y - output.squeeze())**2).mean() #loss(y, output.squeeze())

    return
        
def DI_two_columns(S, Y):
    PY1S1, PY1S0 = Y[S==1].mean(), Y[S==0].mean()
    return min(PY1S1,PY1S0) / max(PY1S1,PY1S0)

def EoO_3_columns(S, pred, Y):
    S, pred = S[Y==1], pred[Y==1]
    PY1S1, PY1S0 = pred[S==1].mean(), pred[S==0].mean()
    return abs(PY1S1 - PY1S0)

def calculate_DI_bins(bins):
    n1, n0 = bins[0] + bins[1], bins[2] + bins[3]
    PY1S1, PY1S0 = bins[0]/n1, bins[2] / n0
    DI =  PY1S0 / PY1S1
    return DI

def calculate_DP_bins(bins):
    n1, n0 = bins[0] + bins[1], bins[2] + bins[3]
    PY1S1, PY1S0 = bins[0]/n1, bins[2] / n0
    DP =  PY1S1 - PY1S0
    return DP

def DI_fct(arr, S_index, Y_index):
    return calculate_DI_bins(transform_arr_to_bins_exact(arr, S_index = S_index, Y_index = Y_index))
    
def check_orthogonal(a, tol=1e-6):
    n = a.shape[0]
    return np.all(np.abs(a @ a.T - np.eye(n,n)) < tol)

def matrix_square_root(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    if check_orthogonal(matrix):    
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    else:
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ np.linalg.inv(eigvecs)
    
def DI_fct_dataset(df, Y_name):
    PY1S1 = (df[(df.S == 1) & (df[Y_name] == 1)].shape[0]/df[df.S == 1].shape[0]) 
    PY1S0 = (df[(df.S == 0) & (df[Y_name] == 1)].shape[0]/df[df.S == 0].shape[0])
    return min(PY1S1,PY1S0) / max(PY1S1,PY1S0)

    
def DI_S0_discriminated(df, Y_name):
    PY1S1 = (df[(df.S == 1) & (df[Y_name] == 1)].shape[0]/df[df.S == 1].shape[0]) 
    PY1S0 = (df[(df.S == 0) & (df[Y_name] == 1)].shape[0]/df[df.S == 0].shape[0])
    return min(PY1S1,PY1S0) == PY1S0

def verify_enought_ind(df, S_name, Y_name, num_ind):
    n = df.shape[0]
    df_copy = df[[S_name, Y_name]].copy()
    return ((df_copy.value_counts()<num_ind).sum() == 0) and ((df_copy.value_counts() > (n - num_ind)).sum() == 0)

def transform_arr_to_bins_exact(arr,
                          S_index,
                          Y_index):
    '''
    array with S and Y column, count the bins of them in this order:  
    Y1S1, Y0S1, Y1S0, Y0S0
    '''
    Y1S1 = np.sum(arr[:,S_index] * arr[:,Y_index])
    Y0S1 = np.sum(arr[:,S_index] * (1 - arr[:,Y_index])) 
    Y1S0 = np.sum((1 - arr[:,S_index]) * arr[:,Y_index]) 
    Y0S0 = np.sum((1 - arr[:,S_index]) * (1 - arr[:,Y_index])) 

    bins = np.array([int(Y1S1), int(Y0S1), int(Y1S0), int(Y0S0)], dtype = int)
    return bins

def transform_arr_to_bins_lambda(arr,
                          S_index,
                          Y_index,
                          L_index):
    '''
    array with S and Y column, count the bins of them in this order:  
    Y1S1, Y0S1, Y1S0, Y0S0
    '''
    n = len(arr)
    arr[:,L_index] = n * arr[:,L_index]
    Y1S1 = np.sum(arr[:,S_index] * arr[:,Y_index] * arr[:,L_index])
    Y0S1 = np.sum(arr[:,S_index] * (1 - arr[:,Y_index]) * arr[:,L_index]) 
    Y1S0 = np.sum((1 - arr[:,S_index]) * arr[:,Y_index] * arr[:,L_index]) 
    Y0S0 = np.sum((1 - arr[:,S_index]) * (1 - arr[:,Y_index]) * arr[:,L_index])
    bins = np.array([int(Y1S1//1), int(Y0S1//1), int(Y1S0//1), int(Y0S0//1)], dtype = int)
    remainders = np.array([Y1S1 % 1, Y0S1 % 1, Y1S0 % 1, Y0S0 % 1])
    while bins.sum() < n:
        j = np.argmax(remainders)
        bins[j] += 1
        remainders[j] = 0
    return bins

def transform_arr_to_bins(arr,
                          S_index,
                          Y_index,
                          L_index = None):
    '''
    array with S and Y column, count the bins of them in this order:  
    Y1S1, Y0S1, Y1S0, Y0S0
    '''
    if L_index is None:
        bins =  transform_arr_to_bins_exact(arr,
                                           S_index,
                                           Y_index)
    else:
        bins =  transform_arr_to_bins_lambda(arr,
                                            S_index,
                                            Y_index,
                                            L_index)
    return bins

#From Gems dataframes.py

def Transform_df_categories(input_df,cols_categorical):
    """
    Transform all categorial variables into integer values and get the convertion to the original name.

    Input:
    - input_df: the studied pandas dataframe
    - cols_categorical: list containing the name of the categorical variables
    Ouptuts:
    - output_df: the transformed dataframe
    - Categories_name_to_id: dictionary containing the conversion from the categories of input_df to those of output_df
    - Categories_id_to_name: dictionary containing the conversion from the categories of output_df to those of input_df
    """

    cols_categorical = []
    for column in input_df.columns:
        if is_string_dtype(input_df[column]):
            cols_categorical.append(column)

    output_df=input_df.copy(deep=True)

    Categories_name_to_id={}
    Categories_id_to_name={}

    #replace the categories by integer labels
    for variable in cols_categorical:
        Categories_name_to_id[variable]={}
        Categories_id_to_name[variable]={}

        list_categories=list(set(output_df[variable]))
        list_categories.sort()
        id_categories=list(range(len(list_categories)))
        #print('For variable',variable,':',list_categories,' -> ',id_categories)

        output_df[variable].replace(list_categories,id_categories, inplace=True)

        for i in range(len(list_categories)):
            Categories_name_to_id[variable][list_categories[i]]=id_categories[i]
            Categories_id_to_name[variable][id_categories[i]]=list_categories[i]

    return output_df,Categories_name_to_id,Categories_id_to_name