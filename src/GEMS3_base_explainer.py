import numpy as np


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+  obs_stresser: Main class to stress the observations (other classes use it)  +
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class obs_stresser:

    """
    Class to stress a set of observations. The observations are set when initializing the class.
    Stress type and stress parameters will be set when running the fit method
    """

    def __create_column_var__(self, id_column, mean):
        column = ((self.observations[:,id_column] - mean)**2).reshape(-1,1) #reshape in order to concatenate
        self.observations = np.concatenate((self.observations, column), axis=1)
        return

    def __create_column_cov__(self, id_column_1, id_column_2, mean_1, mean_2):
        column = ((self.observations[:,id_column_1] - mean_1) * (self.observations[:,id_column_2] - mean_2)).reshape(-1,1)
        self.observations = np.concatenate((self.observations, column), axis=1)
        return

    def __transform_value_DI__(self, wanted_value, id_column_Ypred, id_column_S):
        Ypred = self.observations[:,id_column_Ypred]
        S = self.observations[:,id_column_S]
        n1 = np.sum(self.observations[:,id_column_S])
        return (wanted_value - 1)*np.sum((Ypred * S)/n1)

    def __create_column_DI__(self, id_column_Ypred, id_column_S):
        Ypred = self.observations[:,id_column_Ypred]
        S = self.observations[:,id_column_S]
        n1 = np.sum(self.observations[:,id_column_S])
        column =  ( self.n * ( (Ypred * (1-S))/( self.n - n1)  - (Ypred * S)/n1 ) ).reshape(-1,1)
        self.observations = np.concatenate((self.observations, column), axis=1)
        return

    def __init_if_not__(self, dic, id_column):
        if not id_column in dic['means']:
            dic['means'][id_column] = np.mean(self.observations[:,id_column])
        return dic


    def __init__(self, observations):
        """
        observations: Matrix containing the reference observations (Each row
          represents an observation and each column a variable)
        """

        self.observations=observations

        [self.n,self.p]=observations.shape

        #print("The observation matrix contains:")
        #print("   "+str(self.n)+" observations")
        #print("   "+str(self.p)+" variables")

        self.ksi=np.zeros([1])
        self.ksi_is_computed=False



    def __call__(self, ksis):
        """
        Return the value of the obective function

        ksis is a np.array vector containing the weights on the different
        conditions. Its size must be equal to self.nb_conditions as its values
        are related to self.phi and self.targets
        """

        Z_ksis=np.mean(np.exp(np.dot(self.phi,ksis)))
        H_ksis=np.log(Z_ksis)-np.dot(ksis,self.targets)

        return H_ksis

    def __cpt_grad_H_ksis(self, ksis):
        """
        return the gradient of the objective function wrt the vector ksis
        """

        exp_phi_ksi=np.exp(np.dot(self.phi,ksis))

        grad_H_ksis=-self.targets

        numerator=grad_H_ksis*0.
        for i in range(len(numerator)):
            numerator[i]=np.sum((exp_phi_ksi.flatten())*(self.phi[:,i].flatten()))
        denominator=np.sum(exp_phi_ksi)

        grad_H_ksis+=(numerator/denominator)

        return grad_H_ksis

    def _cpt_ksis_standard_gd(self,gd_iterations,lr,lr_fixed):
        """
        Perform a standard gradient descent to estimate ksis
        """

        tst_ksi=np.zeros([self.nb_conditions])
        if not lr_fixed:
            value = self.__call__(tst_ksi)

        try:   #NEW 11/02
            for i in range(gd_iterations):
                grad_H_ksis=self.__cpt_grad_H_ksis(tst_ksi)
                tst_ksi-=lr*grad_H_ksis
                if not lr_fixed:
                    new_value = self.__call__(tst_ksi)
                    if new_value > value:
                        lr = lr / 2
                        tst_ksi += lr*grad_H_ksis
                        new_value = self.__call__(tst_ksi)
                    value = new_value
        except:   #NEW 11/02
            print("Warning: An error occured during the gradient descent")   #NEW 11/02

        return tst_ksi



    def fit(self, dic,lr=0.1,gd_iterations=20,verbose=False, lr_fixed = False):
        """
        Compute the optimal np.array vector of weights on the conditions (i.e. ksi).
        A simple gradient descent is used to find the optimal ksi.
        Input parameters:
            - dic: a dictionnary which can have 5 different keys:
                - 'means'      : {id_column_1: value_1 , ... , id_column_n : value_n }
                - 'var'   : [id_column_1, value_1 , ... , id_column_n , value_n ]
                - 'cov' : [id_column_1, id_column_2, value_1 , ..., id_column_2n-1, id_column_2n, value_n]
                - 'DI'       : [id_column_Ypred, id_column_S, value]
            - lr: the learning rate of the gradient descent
            - gd_iterations: number of gradient descent iterations
            - lr_fixed: if the learning rate is fixed or can decreased during the descent
            - verbose: The accuracy of the result is shown if verbose==True
        """
        assert ('DI_n1_bloqued' not in dic) or ('DI_while' not in dic), 'must use at most one DI criteria'

        self.observations=self.observations[:,:(self.p)] #If a fit was done previously this removes the columns added.

        if not 'means' in dic:
            dic['means'] = {}
        if 'var' in dic:
            List_stress_var = dic['var']
            for i in range (int(len(List_stress_var)/2)):
                dic['means'][self.observations.shape[1]] = float(List_stress_var[i+1])
                dic = self.__init_if_not__(dic, List_stress_var[i])
                self.__create_column_var__(List_stress_var[i], dic['means'][List_stress_var[i]])
        if 'cov' in dic:
            List_stress_cov = dic['cov']
            for i in range (int(len(List_stress_cov)/3)):
                dic['means'][self.observations.shape[1]] = float(List_stress_cov[i+2])
                dic = self.__init_if_not__(dic, List_stress_cov[i])
                dic = self.__init_if_not__(dic, List_stress_cov[i+1])
                id_col_1, id_col_2 = List_stress_cov[i], List_stress_cov[i+1]
                self.__create_column_cov__(id_col_1, id_col_2, dic['means'][id_col_1], dic['means'][id_col_2] )

        if 'DI' in dic: #WantedChanged, n1 bloqued
            if verbose:
                print('We strongly recommend using \'obs_stresser_fairness\' to stress a disparate impact. The results will be more accurate.')
            List_stress_DI = dic['DI']
            dic['means'][self.observations.shape[1]] = self.__transform_value_DI__(List_stress_DI[2], List_stress_DI[0], List_stress_DI[1])
            dic = self.__init_if_not__(dic, List_stress_DI[1])
            self.__create_column_DI__(List_stress_DI[0], List_stress_DI[1])


        self.dic = dic
        #1) init
        self.nb_conditions=len(dic['means'])
        self.phi=np.zeros([self.n,self.nb_conditions])
        self.targets=np.zeros([self.nb_conditions])
        for i, key in enumerate(dic['means'].keys()):
            self.phi[:,i]=self.observations[:,key]
            self.targets[i]=dic['means'][key]

        #2) gradient descent
        tst_ksi=self._cpt_ksis_standard_gd(gd_iterations,lr, lr_fixed)

        #3) save the optimal ksi
        self.ksi=tst_ksi.copy()
        self.ksi_is_computed=True

        #4) show the results
        if verbose:
            self.show_results()


        return


    def get_ksi(self):
        """
        return the computed ksis
        """

        if self.ksi_is_computed:
            return self.ksi
        else:
            print("ksi was not computed so far. Please use the \"fit\" method")


    def get_quantile(self,col_id,p):
        """
        Return a quantile of the variable in column \"col_id\" of the
        "observations" matrix. The quantile is returned for a probability \"p\"
        in ]O,1[ (eg, if p=0.5, the median is returned).
        """
        return np.quantile(self.observations[:,col_id], p)

    def print_stats(self,col_id):
        """
        print elementary statistics on the column 'col_id'
        """
        print("min:",np.min(self.observations[:,col_id]))
        print("max:",np.max(self.observations[:,col_id]))
        print("mean:",np.mean(self.observations[:,col_id]))
        print("std:",np.std(self.observations[:,col_id]))

    def get_lambda(self):
        """
        return the weights on the observations for given ksis
        """

        if self.ksi_is_computed:
            lambdas = np.exp(np.dot(self.phi,self.ksi)) / (np.sum(np.exp(np.dot(self.phi,self.ksi))))
            return lambdas
        else:
            print("ksi was not computed so far. Please use the \"fit\" method")

    def get_lambda_no_stress(self):
        """
        return the weights on the observations for given ksis
        """

        if self.ksi_is_computed:
            ksis_no_stress=np.zeros([self.nb_conditions])
            lambdas = np.exp(np.dot(self.phi,ksis_no_stress)) / (np.sum(np.exp(np.dot(self.phi,ksis_no_stress))))
            return lambdas
        else:
            print("ksi was not computed so far. Please use the \"fit\" method")


    def get_stress_impact_on_values(self,values_to_treat):
        """
        Directly compute the weights on the observations for given ksis and apply them to an array of values.
        The mean and a confidence interval of this mean will be returned

        Input:
         - values_to_treat: the values that will be weighted with the weights
        Outputs:
         - mean: mean of the weigthed values
         - mean_Q1: lower confidence interval on this mean (1st decile)
         - mean_Q9: higher confidence interval on this mean (9th decile)
        """

        if self.ksi_is_computed:
            lambdas = np.exp(np.dot(self.phi,self.ksi)) / (np.sum(np.exp(np.dot(self.phi,self.ksi))))
            wvtt_mean,wvtt_std,mean_Q1,mean_Q9=Cpt_mean_std_CImean_of_weighted_obs(lambdas,values_to_treat)
            return wvtt_mean,mean_Q1,mean_Q9
        else:
            print("ksi was not computed so far. Please use the \"fit\" method")
        #compute the mean and std of the weighted data, plus the confidence interval on the means




    def show_results(self):
        """
        Show the results generated by a given vector of ksis
        """

        if self.ksi_is_computed==False:
            print("ksi was not computed so far. Please use the \"fit\" method")
            return 0

        #show the score H
        print('\nksis: ',self.ksi,' -> H = ', self.__call__(self, self.ksi) )

        #show the lambdas
        print('\nlambdas:',self.get_lambda())

        #show the corresponding means, variances or co-variances depending on the stress type
        print('\n')
        for i in range(self.nb_conditions):
            current_mean = np.average(self.phi[:,i].flatten(), weights=self.get_lambda())
            original_mean = np.average(self.phi[:,i].flatten())
            desired_mean=self.targets[i]
            print('Mean of variable '+str(i)+':',current_mean,'  (original='+str(original_mean)+'  desired='+str(desired_mean)+')\n')

    def get_accuracy_score(self):
        """
        Return a distance in which measures whether the input ksi leads to a stress
        on the data close to the desired one.
        -> The closer the score from 1, the more accurate the stress
        -> The closer the score from 0, the closer the stressed data to the
        un-stressed data
        """

        if self.ksi_is_computed==False:
            print("ksi was not computed so far. Please use the \"fit\" method")
            return 0

        #compute the lambdas
        lambdas = self.get_lambda()

        #compute the distance
        distance=0.
        for i in range(self.nb_conditions):
            current_mean = np.average(self.phi[:,i].flatten(), weights=lambdas)
            original_mean = np.average(self.phi[:,i].flatten())
            desired_mean=self.targets[i]
            distance+=(np.abs(current_mean-desired_mean)/np.abs(original_mean-desired_mean))**2
        distance/=self.nb_conditions
        distance=np.sqrt(distance)
        return 1.-distance

    def get_KL_divergence(self):

        if self.ksi_is_computed==False:
            print("ksi was not computed so far. Please use the \"fit\" method")
            return 0

        #compute the lambdas
        lambdas = self.get_lambda()

        #compute the distance
        distance=np.sum(np.multiply(np.log(lambdas * self.n), lambdas))

        return distance



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+  obs_stresser_fairness:                                           +
#+       Directly stresses a fairness index (DI or EOO)              +
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class obs_stresser_fairness:
    """
    Class to stress the data wrt to a fairness indice. This class makes use of the
    fit and get_lambda methods of the obs_stresser class. It stresses the fairness
    indices by defining a pertinent subset of observations and defining an add-oc
    mean stress.
    """

    def __init__(self, 
                 all_observations,
                 Id_col_Ypred,
                 Id_col_S,
                 fairness_type='DI',
                 Id_col_Ytrue=None):
        """
        Class to stress the data wrt to a fairness indice. This class makes use of the
        fit and get_lambda methods of the obs_stresser class. It stresses the fairness
        indices by defining a pertinent subset of observations and defining an add-oc
        mean stress.
        Parameters:
         - all_observations: Matrix containing all reference observations (Each row
                             represents an observation and each column a variable)
         - Id_col_Ypred: column conaining the binary predictions of Y
         - Id_col_S: column containing the binary values S
         - fairness_type: Type of fairness among 'DI', 'DI_S0_only', 'DI_S1_only', 'EOO', 'EOO_S0_only', 'EOO_S1_only'
         - Id_col_Ytrue: column containing the true binary Y (only required for EOO stresses)
        """

        self.ksi=np.zeros([1])
        self.ksi_is_computed=False

        if Id_col_Ytrue is None:
            if fairness_type=='EOO' or fairness_type=='EOO_S0_only'  or fairness_type=='EOO_S1_only':
                print("You must define the column with the true Y values to use an equality of odds metric. The DI is then stressed intead of "+fairness_type+".")
                fairness_type='DI'

        #store general information
        self.all_observations=all_observations
        self.Id_col_Ypred=Id_col_Ypred
        self.Id_col_Ytrue=Id_col_Ytrue
        self.Id_col_S=Id_col_S
        self.fairness_type=fairness_type

        #get the observations into different groups of interest
        self.IDs_obs_of_interest_S0=np.where(self.all_observations[:,self.Id_col_S]<0.5)[0] #==0
        self.IDs_obs_of_interest_S1=np.where(self.all_observations[:,self.Id_col_S]>0.5)[0] #==1
        if Id_col_Ytrue is not None:  #only required by EOO
            self.IDs_obs_of_interest_S0Y1=np.where((self.all_observations[:,self.Id_col_S]<0.5)*(self.all_observations[:,self.Id_col_Ytrue]>0.5))[0] #==0 and ==1
            self.IDs_obs_of_interest_S1Y1=np.where((self.all_observations[:,self.Id_col_S]>0.5)*(self.all_observations[:,self.Id_col_Ytrue]>0.5))[0] #==1 and ==1
            self.IDs_obs_of_interest_Y0=np.where(self.all_observations[:,self.Id_col_Ytrue]<0.5)[0] # ==0

        #get the number of observations into different groups of interest
        self.n=self.all_observations.shape[0]
        self.n_S0=len(self.IDs_obs_of_interest_S0)
        self.n_S1=len(self.IDs_obs_of_interest_S1)
        if self.n!=self.n_S0+self.n_S1:
            print("WARNING: The chosen Id_col_S does not seem to represent a binary sensitive variable")

        if Id_col_Ytrue is not None:
            self.n_S0Y1=len(self.IDs_obs_of_interest_S0Y1)
            self.n_S1Y1=len(self.IDs_obs_of_interest_S1Y1)

        #instanciate the obs_stressers on the pertinent groups of interest
        if self.fairness_type=='DI': #two independent stresses will be made in groups 0 and 1
            self.subsampled_obs_stresser_S0=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S0,:])
            self.subsampled_obs_stresser_S1=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S1,:])
        elif  self.fairness_type=='EOO': #two independent stresses will be made in groups 0 and 1
            self.subsampled_obs_stresser_S0Y1=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S0Y1,:])
            self.subsampled_obs_stresser_S1Y1=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S1Y1,:])
        elif self.fairness_type=='DI_S0_only':
            self.subsampled_obs_stresser_S0=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S0,:])
        elif self.fairness_type=='DI_S1_only':
            self.subsampled_obs_stresser_S1=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S1,:])
        elif self.fairness_type=='EOO_S0_only':
            self.subsampled_obs_stresser_S0Y1=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S0Y1,:])
        elif self.fairness_type=='EOO_S1_only':
            self.subsampled_obs_stresser_S1Y1=obs_stresser(self.all_observations[self.IDs_obs_of_interest_S1Y1,:])
        else:
            print(self.fairness_type+' is an unknown option')


    def get_lambda_unstressed(self):
            """
            Get the values of the lambdas if there is no stress
            """
            if self.fairness_type=='EOO' or self.fairness_type=='EOO_S0_only' or self.fairness_type=='EOO_S1_only': #EOO like stress only applies to the observations having a true Y equal to 1
                unst_lambda_wghts=np.zeros(self.n)
                unst_lambda_wghts[self.IDs_obs_of_interest_S0Y1]=1./(self.n_S0Y1+self.n_S1Y1)
                unst_lambda_wghts[self.IDs_obs_of_interest_S1Y1]=1./(self.n_S0Y1+self.n_S1Y1)
            else:
                unst_lambda_wghts=np.ones(self.n)/self.n

            return unst_lambda_wghts

    def get_fairness_type(self):
        return self.fairness_type

    def fit_and_cpt_lambdas(self, 
                            fairness_value,lr=0.1,
                            gd_iterations=20,
                            lr_fixed = False,
                            verbose=False,
                            delta_type = 'mean'):
        """
        Same as fit in the class obs_stresser but only fairness_value is given instead of dic,
        and the lambdas are directly computed.
        Note that depending on the choice of self.fairness_type, the scalar fairness_value will
        be either a desired disparate impact or a desired equality of odds value.
        """
        if self.fairness_type=='DI': #two independent stresses will be made in groups 0 and 1 (delta0 and delta1 are computed so that the level of stress in each group is proportional to the amount of observations in this group)
            #a) compute the proper stress to get using mean positive predictions in a sub-group of observations

            meanY1_S0=np.mean(self.all_observations[self.IDs_obs_of_interest_S0,self.Id_col_Ypred])
            meanY1_S1=np.mean(self.all_observations[self.IDs_obs_of_interest_S1,self.Id_col_Ypred])

            DI = meanY1_S0 / meanY1_S1
            Diff_DI = fairness_value - DI
            lambda_0, lambda_1 = meanY1_S0 * self.n_S0, meanY1_S1 * self.n_S1

            if delta_type == 'mean':
                #Same mean
                delta_1_mean = (lambda_1 / (1 + (1 + (self.n_S1 * lambda_0)/(self.n_S0 * lambda_1)) / Diff_DI) )/ self.n_S1
                delta_0_mean = delta_1_mean
            else:
                #same number of changement
                delta_1 = lambda_1 / (1 + ( self.n_S1 / ( self.n_S0 * Diff_DI) ) * (1 + lambda_0 / lambda_1 ) ) 
                delta_1_mean = delta_1 / self.n_S1
                delta_0_mean = delta_1 / self.n_S0


            '''
            delta1 = -(fairness_value * meanY1_S1 - meanY1_S0 ) / ( fairness_value + (self.n_S0 / self.n_S1))
            delta0 = - delta1 * (self.n_S0 / self.n_S1 )
            '''
            
            dicS0={'means' : {self.Id_col_Ypred: meanY1_S0 + delta_0_mean }}
            dicS1={'means' : {self.Id_col_Ypred: meanY1_S1 - delta_1_mean }}
            if verbose:
                print("MeanPP1 in group S0: "+str(meanY1_S0)+" -> "+str(meanY1_S0+delta_0_mean))
                print("MeanPP1 in group S1: "+str(meanY1_S1)+" -> "+str(meanY1_S1+delta_1_mean))
            #b) stresses
            loc_ksi=self.subsampled_obs_stresser_S0.fit(dicS0 ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            loc_ksi=self.subsampled_obs_stresser_S1.fit(dicS1 ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            #c) recompose the lambda values for all observations
            lambda_ooiS0_only=self.subsampled_obs_stresser_S0.get_lambda()
            lambda_ooiS1_only=self.subsampled_obs_stresser_S1.get_lambda()
            self.lambda_wghts=np.zeros(self.n)
            self.lambda_wghts[self.IDs_obs_of_interest_S0]=lambda_ooiS0_only*self.n_S0/(self.n_S0+self.n_S1)  #the sum of lambda_ooi_only was 1
            self.lambda_wghts[self.IDs_obs_of_interest_S1]=lambda_ooiS1_only*self.n_S1/(self.n_S0+self.n_S1)  #the sum of lambda_ooi_only was 1
        elif self.fairness_type=='DI_S0_only':
            #a) compute the proper stress to get using mean positive predictions in a sub-group of observations
            meanY1_S1=np.mean(self.all_observations[self.IDs_obs_of_interest_S1,self.Id_col_Ypred])
            dic={'means' : {self.Id_col_Ypred: fairness_value*meanY1_S1 }}
            #b) stress
            loc_ksi=self.subsampled_obs_stresser_S0.fit(dic ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            #c) recompose the lambda values for all observations
            lambda_ooi_only=self.subsampled_obs_stresser_S0.get_lambda()
            self.lambda_wghts=np.zeros(self.n)
            self.lambda_wghts[self.IDs_obs_of_interest_S0]=lambda_ooi_only*self.n_S0/(self.n_S0+self.n_S1)  #the sum of lambda_ooi_only was 1
            self.lambda_wghts[self.IDs_obs_of_interest_S1]=1./(self.n_S0+self.n_S1)
        elif self.fairness_type=='DI_S1_only':
            #a) compute the proper stress to get using mean positive predictions in a sub-group of observations
            meanY1_S0=np.mean(self.all_observations[self.IDs_obs_of_interest_S0,self.Id_col_Ypred])
            dic={'means' : {self.Id_col_Ypred: meanY1_S0/fairness_value }}
            #b) stress
            loc_ksi=self.subsampled_obs_stresser_S1.fit(dic ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            #c) recompose the lambda values for all observations
            lambda_ooi_only=self.subsampled_obs_stresser_S1.get_lambda()
            self.lambda_wghts=np.zeros(self.n)
            self.lambda_wghts[self.IDs_obs_of_interest_S0]=1./(self.n_S0+self.n_S1)
            self.lambda_wghts[self.IDs_obs_of_interest_S1]=lambda_ooi_only*self.n_S1/(self.n_S0+self.n_S1)  #the sum of lambda_ooi_only was 1
        elif  self.fairness_type=='EOO': #two independent stresses will be made in groups 0 and 1
            #a) compute the proper stress to get using mean positive predictions in a sub-group of observations
            meanY1_S0Y1=np.mean(self.all_observations[self.IDs_obs_of_interest_S0Y1,self.Id_col_Ypred])
            meanY1_S1Y1=np.mean(self.all_observations[self.IDs_obs_of_interest_S1Y1,self.Id_col_Ypred])
            delta1=-(fairness_value*meanY1_S1Y1-meanY1_S0Y1)/(fairness_value+(self.n_S0Y1/self.n_S1Y1))
            delta0=-delta1*(self.n_S0Y1/self.n_S1Y1)
            dicS0={'means' : {self.Id_col_Ypred: meanY1_S0Y1+delta0 }}
            dicS1={'means' : {self.Id_col_Ypred: meanY1_S1Y1+delta1 }}
            if verbose:
                print("MeanPP1 in group 0 with true Y=1: "+str(meanY1_S0Y1)+" -> "+str(meanY1_S0Y1+delta0))
                print("MeanPP1 in group 1 with true Y=1: "+str(meanY1_S1Y1)+" -> "+str(meanY1_S1Y1+delta1))
            #b) stresses
            loc_ksi=self.subsampled_obs_stresser_S0Y1.fit(dicS0 ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            loc_ksi=self.subsampled_obs_stresser_S1Y1.fit(dicS1 ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            #c) recompose the lambda values for all observations
            lambda_ooiS0Y1_only=self.subsampled_obs_stresser_S0Y1.get_lambda()
            lambda_ooiS1Y1_only=self.subsampled_obs_stresser_S1Y1.get_lambda()
            self.lambda_wghts=np.zeros(self.n)
            self.lambda_wghts[self.IDs_obs_of_interest_S0Y1]=lambda_ooiS0Y1_only*self.n_S0Y1/(self.n_S0Y1+self.n_S1Y1)  #the sum of lambda_ooi_only was 1
            self.lambda_wghts[self.IDs_obs_of_interest_S1Y1]=lambda_ooiS1Y1_only*self.n_S1Y1/(self.n_S0Y1+self.n_S1Y1)  #the sum of lambda_ooi_only was 1
        elif self.fairness_type=='EOO_S0_only':
            #a) compute the proper stress to get using mean positive predictions in a sub-group of observations
            meanY1_S1Y1=np.mean(self.all_observations[self.IDs_obs_of_interest_S1Y1,self.Id_col_Ypred])
            dic={'means' : {self.Id_col_Ypred: fairness_value*meanY1_S1Y1 }}
            #b) stress
            loc_ksi=self.subsampled_obs_stresser_S0Y1.fit(dic ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            #c) recompose the lambda values for all observations
            lambda_ooi_only=self.subsampled_obs_stresser_S0Y1.get_lambda()
            self.lambda_wghts=np.zeros(self.n)
            self.lambda_wghts[self.IDs_obs_of_interest_S0Y1]=lambda_ooi_only*self.n_S0Y1/(self.n_S0Y1+self.n_S1Y1)  #the sum of lambda_ooi_only was 1
            self.lambda_wghts[self.IDs_obs_of_interest_S1Y1]=1./(self.n_S0Y1+self.n_S1Y1)  #the sum of lambda_ooi_only was 1
        elif self.fairness_type=='EOO_S1_only':
            #a) compute the proper stress to get using mean positive predictions in a sub-group of observations
            meanY1_S0Y1=np.mean(self.all_observations[self.IDs_obs_of_interest_S0Y1,self.Id_col_Ypred])
            dic={'means' : {self.Id_col_Ypred: meanY1_S0Y1/fairness_value }}
            #b) stress
            loc_ksi=self.subsampled_obs_stresser_S1Y1.fit(dic ,lr=lr,gd_iterations=gd_iterations,lr_fixed = lr_fixed,verbose=verbose)
            #c) recompose the lambda values for all observations
            lambda_ooi_only=self.subsampled_obs_stresser_S1Y1.get_lambda()
            self.lambda_wghts=np.zeros(self.n)
            self.lambda_wghts[self.IDs_obs_of_interest_S0Y1]=1./(self.n_S0Y1+self.n_S1Y1)  #the sum of lambda_ooi_only was 1
            self.lambda_wghts[self.IDs_obs_of_interest_S1Y1]=lambda_ooi_only*self.n_S1Y1/(self.n_S0Y1+self.n_S1Y1)  #the sum of lambda_ooi_only was 1
        else:
            print('I said '+self.fairness_type+' is an unknown option')
        self.ksi_is_computed=True
        self.ksi = loc_ksi


    def get_lambda(self):
        """
        Same as get_lambda in the class obs_stresser but only for the observations self.IDs_obs_of_interest.
        The values of lambda for the observations which are not of interest will be set to zero.
        """
        return self.lambda_wghts
    
    def get_KL_divergence(self):

        if self.ksi_is_computed==False:
            print("ksi was not computed so far. Please use the \"fit\" method")
            return 0

        #compute the lambdas
        lambdas = self.get_lambda()

        #compute the distance
        distance=np.sum(np.multiply(np.log(lambdas * self.n), lambdas))

        return distance

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+  Functions to measure the impact of  stresses (may be in another file) +
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_quantile(Data,col_name,value):
    """
    Get the quantile of 'value' in the column 'col_name' of the dataframe 'Data'
    """
    q=np.arange(1000)/1000
    qvals=np.quantile(Data[col_name],q)
    i=0
    while value>qvals[i] and i<999:
        i+=1
    return q[i]

def CompareStressImpacts(Data,lambdas1,lambdas2,thresh_binary_var=1.):
    """
    Compare the impact of different stresses on the variables composing the
    dataframe 'Data'. These lambdas can be for instance those obtained with
    a stress and no stress. They can also be obtained by stressing S0 only
    and S1 only.
    Remarks:
    - 'lambdas1' and 'lambdas2' must have the size as the amount of
    observations in Data, as they represent the weights associated to each observation.
    - Binary and continuous variables are considered. For multi-class variables, please
    use a one-hot-encoding representation of the dataframe
    Options:
    - thresh_binary_var: Only the binary variables with a different of more than
                         'thresh_binary_var' percent will be represented.
    """

    #init
    X_col_names=Data.columns

    BinaryVarTypes=[]
    for X_col_name in X_col_names:
        if len(set(Data[X_col_name]))<3:
            BinaryVarTypes.append(True)
        else:
            BinaryVarTypes.append(False)

    #manage binary variables
    print('\nBinary variables (mean values are represented)')
    for i in range(len(X_col_names)):
        BinaryVarType=BinaryVarTypes[i]
        if BinaryVarType:
            X_col_name=X_col_names[i]
            Stressed_MeanVal1=np.dot(Data[X_col_name],lambdas1)
            Stressed_MeanVal2=np.dot(Data[X_col_name],lambdas2)
            if np.fabs(Stressed_MeanVal1-Stressed_MeanVal2)*100.>thresh_binary_var:
                print(X_col_name+': ',np.round(Stressed_MeanVal1,3),np.round(Stressed_MeanVal2,3),' -> Diff='+str(np.round(100*(Stressed_MeanVal1-Stressed_MeanVal2),1))+'%')


    #manage non-binary variables
    print('\nNon binary variables (quantiles of the mean values are represented)')
    for i in range(len(X_col_names)):
        BinaryVarType=BinaryVarTypes[i]
        if not BinaryVarType:
            X_col_name=X_col_names[i]
            Stressed_MeanVal1=np.dot(Data[X_col_name],lambdas1)
            Stressed_MeanVal2=np.dot(Data[X_col_name],lambdas2)
            q_Stressed_MeanVal1=get_quantile(Data,X_col_name,Stressed_MeanVal1)
            q_Stressed_MeanVal2=get_quantile(Data,X_col_name,Stressed_MeanVal2)
            print(X_col_name+': ',np.round(q_Stressed_MeanVal1,3),np.round(q_Stressed_MeanVal2,3),' -> Diff='+str(np.round(q_Stressed_MeanVal1-q_Stressed_MeanVal2,3)))





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+        FUNCTIONS FOR FAIRNESS AND UNCERTAINTY QUANTIFICATION           +
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Cpt_mean_std_CImean_of_weighted_obs(weigths,observations):
    """
    Return the mean, the standard deviation and a confidence interval (1st and 9th deciles) of the true mean (with an inifinite number of observations), on weighted values
    Inputs:
      - weigths: np.array with the weights
      - observations: np.array containing the orginial values to average
    Outputs:
      - mean: float
      - std: float
      - ICmean: two floats
    """

    #compute the mean and std of the weighted data
    n=weigths.shape[0]
    sumweigths=weigths.sum()

    weigths_tilde=weigths*n/sumweigths   #we want the average lambda value to be 1

    weighted_observations=weigths_tilde*observations

    wvtt_mean=weighted_observations.mean()
    wvtt_std=weighted_observations.std()

    #compute the confidence interval
    scaling_factor=1.96/np.sqrt(n)

    Quantiles=[wvtt_mean-(scaling_factor*wvtt_std) , wvtt_mean+(scaling_factor*wvtt_std)]

    return wvtt_mean,wvtt_std,Quantiles[0],Quantiles[1]




def Cpt_fairness_indicators_of_weigthed_observations(in_S, in_y_pred, in_y_true, in_weights) :
    """
    Return standard fairness indicators and their confidence interval for weighted observations
    - S contains the values of the sensitive variable. Its values must be 0 or 1.
    - y_pred contains the binary predictions. Its values must be 0 or 1.
    - y_true contains the expected (true) binary predictions. Its values must be 0 or 1.
    - weights contains the weights given to each observation
    """

    S=in_S.flatten()
    y_pred=in_y_pred.flatten()
    y_true=in_y_true.flatten()
    weights=in_weights.flatten()
    results={}

    #1) disparate impact...
    #1.1) precomputations
    locS1=np.where(S>0.5)[0]
    n_S1=len(locS1)
    av__Yp1_if_S1,std__Yp1_if_S1,mQ1__Yp1_if_S1,mQ9__Yp1_if_S1=Cpt_mean_std_CImean_of_weighted_obs(weights[locS1],y_pred[locS1])

    locS0=np.where(S<=0.5)[0]
    n_S0=len(locS0)
    av__Yp1_if_S0,std__Yp1_if_S0,mQ1__Yp1_if_S0,mQ9__Yp1_if_S0=Cpt_mean_std_CImean_of_weighted_obs(weights[locS0],y_pred[locS0])

    #1.2) value
    DI=av__Yp1_if_S0/av__Yp1_if_S1

    #1.3) confidence interval
    if av__Yp1_if_S1>0.:
        sigma_loc=np.sqrt( np.power(std__Yp1_if_S0,2)/np.power(av__Yp1_if_S1,2) + np.power(av__Yp1_if_S0,2)*np.power(std__Yp1_if_S1,2)/np.power(av__Yp1_if_S1,4) )
        DI_CI=[DI-1.96*sigma_loc/np.sqrt(n_S0+n_S1) , DI+1.96*sigma_loc/np.sqrt(n_S0+n_S1)]
    else:
        DI_CI=[DI,DI]

    #1.4) store results
    results['DI']=DI.item()
    results['DI_CI_Q1']=DI_CI[0].item()
    results['DI_CI_Q9']=DI_CI[1].item()


    #2) equality of odds...
    #2.1) precomputations
    locS1=np.where((S>0.5) & (y_true>0.5))[0]
    n_S1=len(locS1)
    av__Yp1_if_S1_Yt1,std__Yp1_if_S1_Yt1,mQ1__Yp1_if_S1_Yt1,mQ9__Yp1_if_S1_Yt1=Cpt_mean_std_CImean_of_weighted_obs(weights[locS1],y_pred[locS1])

    locS0=np.where((S<=0.5) & (y_true>0.5))[0]
    n_S0=len(locS0)
    av__Yp1_if_S0_Yt1,std__Yp1_if_S0_Yt1,mQ1__Yp1_if_S0_Yt1,mQ9__Yp1_if_S0_Yt1=Cpt_mean_std_CImean_of_weighted_obs(weights[locS0],y_pred[locS0])

    #2.2) value
    EOD=av__Yp1_if_S0_Yt1/av__Yp1_if_S1_Yt1

    #2.3) confidence interval
    if av__Yp1_if_S1_Yt1>0.:
        sigma_loc=np.sqrt( np.power(std__Yp1_if_S0_Yt1,2)/np.power(av__Yp1_if_S1_Yt1,2) + np.power(av__Yp1_if_S0_Yt1,2)*np.power(std__Yp1_if_S1_Yt1,2)/np.power(av__Yp1_if_S1_Yt1,4) )
        EOD_CI=[EOD-1.96*sigma_loc/np.sqrt(n_S0+n_S1) , EOD+1.96*sigma_loc/np.sqrt(n_S0+n_S1)]
    else:
        EOD_CI=[EOD,EOD]

    #2.4) store results
    results['EOD']=EOD.item()
    results['EOD_CI_Q1']=EOD_CI[0].item()
    results['EOD_CI_Q9']=EOD_CI[1].item()

    #3) equality of opportunities...
    #3.1) precomputations
    locS1=np.where((S>0.5) & (y_true<=0.5))[0]
    n_S1=len(locS1)
    av__Yp1_if_S1_Yt0,std__Yp1_if_S1_Yt0,mQ1__Yp1_if_S1_Yt0,mQ9__Yp1_if_S1_Yt0=Cpt_mean_std_CImean_of_weighted_obs(weights[locS1],y_pred[locS1])

    locS0=np.where((S<=0.5) & (y_true<=0.5))[0]
    n_S0=len(locS0)
    av__Yp1_if_S0_Yt0,std__Yp1_if_S0_Yt0,mQ1__Yp1_if_S0_Yt0,mQ9__Yp1_if_S0_Yt0=Cpt_mean_std_CImean_of_weighted_obs(weights[locS0],y_pred[locS0])

    #3.2) value
    EOP=av__Yp1_if_S0_Yt0/av__Yp1_if_S1_Yt0

    #3.3) confidence interval
    if av__Yp1_if_S1_Yt0>0.:
        sigma_loc=np.sqrt( np.power(std__Yp1_if_S0_Yt0,2)/np.power(av__Yp1_if_S1_Yt0,2) + np.power(av__Yp1_if_S0_Yt0,2)*np.power(std__Yp1_if_S1_Yt0,2)/np.power(av__Yp1_if_S1_Yt0,4) )
        EOP_CI=[EOP-1.96*sigma_loc/np.sqrt(n_S0+n_S1) , EOP+1.96*sigma_loc/np.sqrt(n_S0+n_S1)]
    else:
        EOP_CI=[EOP,EOP]

    #3.4) store results
    results['EOP']=EOP.item()
    results['EOP_CI_Q1']=EOP_CI[0].item()
    results['EOP_CI_Q9']=EOP_CI[1].item()

    #4) TPR and TNR for S=1 and S=0...
    #4.1) value
    TPR_S1=av__Yp1_if_S1_Yt1
    TPR_S0=av__Yp1_if_S0_Yt1
    TNR_S1=1.-av__Yp1_if_S1_Yt0
    TNR_S0=1.-av__Yp1_if_S0_Yt0

    #4.2) confidence interval
    TPR_S1_CI=[mQ1__Yp1_if_S1_Yt1,mQ9__Yp1_if_S1_Yt1]
    TPR_S0_CI=[mQ1__Yp1_if_S0_Yt1,mQ9__Yp1_if_S0_Yt1]
    TNR_S1_CI=[1.-mQ9__Yp1_if_S1_Yt0,1.-mQ1__Yp1_if_S1_Yt0]
    TNR_S0_CI=[1.-mQ9__Yp1_if_S0_Yt0,1.-mQ1__Yp1_if_S0_Yt0]

    #4.3) store results
    results['TPR_S0']=TPR_S0.item()
    results['TPR_S0_CI_Q1']=TPR_S0_CI[0].item()
    results['TPR_S0_CI_Q9']=TPR_S0_CI[1].item()

    results['TPR_S1']=TPR_S1.item()
    results['TPR_S1_CI_Q1']=TPR_S1_CI[0].item()
    results['TPR_S1_CI_Q9']=TPR_S1_CI[1].item()

    results['TNR_S0']=TNR_S0.item()
    results['TNR_S0_CI_Q1']=TNR_S0_CI[0].item()
    results['TNR_S0_CI_Q9']=TNR_S0_CI[1].item()

    results['TNR_S1']=TNR_S1.item()
    results['TNR_S1_CI_Q1']=TNR_S1_CI[0].item()
    results['TNR_S1_CI_Q9']=TNR_S1_CI[1].item()

    return results


def Cpt_disparate_impact_of_weigthed_observations(in_S, in_y_pred, in_weights):
    """

    Truncated version of Cpt_fairness_indicators_of_weigthed_observations with the disparate
    impact only. Should be used when in_y_true is unknown

    Return the disparate impact and its confidence interval for weighted observations
    - S contains the values of the sensitive variable. Its values must be 0 or 1.
    - y_pred contains the binary predictions. Its values must be 0 or 1.
    - weights contains the weights given to each observation
    """

    S=in_S.flatten()
    y_pred=in_y_pred.flatten()
    weights=in_weights.flatten()
    results={}

    #1) disparate impact...
    #1.1) precomputations
    locS1=np.where(S>0.5)[0]
    n_S1=len(locS1)
    av__Yp1_if_S1,std__Yp1_if_S1,mQ1__Yp1_if_S1,mQ9__Yp1_if_S1=Cpt_mean_std_CImean_of_weighted_obs(weights[locS1],y_pred[locS1])

    locS0=np.where(S<=0.5)[0]
    n_S0=len(locS0)
    av__Yp1_if_S0,std__Yp1_if_S0,mQ1__Yp1_if_S0,mQ9__Yp1_if_S0=Cpt_mean_std_CImean_of_weighted_obs(weights[locS0],y_pred[locS0])

    #1.2) value
    DI=av__Yp1_if_S0/av__Yp1_if_S1

    #1.3) confidence interval
    if av__Yp1_if_S1>0.:
        sigma_loc=np.sqrt( np.power(std__Yp1_if_S0,2)/np.power(av__Yp1_if_S1,2) + np.power(av__Yp1_if_S0,2)*np.power(std__Yp1_if_S1,2)/np.power(av__Yp1_if_S1,4) )
        DI_CI=[DI-1.96*sigma_loc/np.sqrt(n_S0+n_S1) , DI+1.96*sigma_loc/np.sqrt(n_S0+n_S1)]
    else:
        DI_CI=[DI,DI]

    #1.4) store results
    results['DI']=DI.item()
    results['DI_CI_Q1']=DI_CI[0].item()
    results['DI_CI_Q9']=DI_CI[1].item()

    return results
