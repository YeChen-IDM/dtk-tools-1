# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:53:08 2017

@author: TingYu Ho
This function uniformly sampling i_n_samp sample points with i_n_rep replication in the subregions c_subregion
input:
    i_n_samp: # sampling needed in the subregion
    i_n_rep: # replication needed in the subregion
    c_subr:examing subregion
    f_alpha
outout
    c_subr:update subregions
"""

def fun_uni_sampler(c_subr, f_alpha, i_n_samp, i_n_rep, i_dim):
    import numpy as np
    import pandas as pd
    import copy
    
    def turn_to_power(list, power=1): 
        return [number**power for number in list]
    
    c_subr.pd_sample_record=c_subr.pd_sample_record.sort_values (by="mean", ascending=True) # sort before start
    c_subr.pd_sample_record=c_subr.pd_sample_record.reset_index(drop=True) # reindex before start
    if len(c_subr.pd_sample_record) >= i_n_samp: # if has enough number of sampling points
        for i in (i for i in range(0,len(c_subr.pd_sample_record)) if c_subr.pd_sample_record.loc[i,'# rep'] < i_n_rep): # check enough # reps or not, i is index
            #for j in range(int(record.loc[i,'# rep']),i_n_rep): # counting number of replications         
            l_output = fun_blackbox(np.array(c_subr.pd_sample_record.loc[i,'x1':'x'+str(i_dim)]), i_n_rep-int(c_subr.pd_sample_record.loc[i,'# rep'])) 
            c_subr.pd_sample_record.loc[i,'mean'] = copy.copy(float(int(c_subr.pd_sample_record.loc[i,'# rep'])*c_subr.pd_sample_record.loc[i,'mean']+sum(l_output))/i_n_rep)
            c_subr.pd_sample_record.loc[i,'SST'] = copy.copy(c_subr.pd_sample_record.loc[i,'SST']+sum(turn_to_power(l_output,2)))
            c_subr.pd_sample_record.loc[i,'var'] = copy.copy(float(c_subr.pd_sample_record.loc[i,'SST']-i_n_rep*pow(c_subr.pd_sample_record.loc[i,'mean'],2))/(i_n_rep-1))
            c_subr.pd_sample_record.loc[i,'# rep'] = i_n_rep

    elif len(c_subr.pd_sample_record) < i_n_samp: # if has not enough sampling points and replication
        if len(c_subr.pd_sample_record)>=1: # if already has sample points, first deal with them
            for i in (i for i in range(0,len(c_subr.pd_sample_record)) if c_subr.pd_sample_record.loc[i,'# rep'] < i_n_rep):# check enough # reps or not for existing old sampling points
                l_output = fun_blackbox(np.array(c_subr.pd_sample_record.loc[i,'x1':'x'+str(i_dim)]), i_n_rep-int(c_subr.pd_sample_record.loc[i,'# rep'])) 
                c_subr.pd_sample_record.loc[i,'mean'] = copy.copy(float(int(c_subr.pd_sample_record.loc[i,'# rep'])*c_subr.pd_sample_record.loc[i,'mean']+sum(l_output))/i_n_rep)
                c_subr.pd_sample_record.loc[i,'SST'] = copy.copy(c_subr.pd_sample_record.loc[i,'SST']+sum(turn_to_power(l_output,2)))
                c_subr.pd_sample_record.loc[i,'var'] = copy.copy(float(c_subr.pd_sample_record.loc[i,'SST']-i_n_rep*pow(c_subr.pd_sample_record.loc[i,'mean'],2))/(i_n_rep-1))
                c_subr.pd_sample_record.loc[i,'# rep'] = i_n_rep
                                           
        i_ini_length=len(c_subr.pd_sample_record) # number of sampling point sofar in this subregion

        for i in range (i_ini_length,i_n_samp): # create new rows for new sampling points
            c_subr.pd_sample_record.loc[i] = [1 for n in range(len(c_subr.pd_sample_record.columns))]
        for i in range (0, i_dim): #create new sampling new point and add new sample point to dataframe
            a_new_sample = np.random.uniform(low=c_subr.l_coordinate_lower[i],high=c_subr.l_coordinate_upper[i], size=i_n_samp-i_ini_length) # generate only for one dim
            index = [x for x in range (i_ini_length,i_n_samp)]
            c_subr.pd_sample_record.loc[i_ini_length:i_n_samp-1, 'x'+str(i+1)]=pd.Series(a_new_sample.tolist(), index)
            #c_subr.pd_sample_record.loc[index,'x'+str(i+1)]=pd.Series(a_new_sample.tolist(),index)                         
        for i in range(i_ini_length,i_n_samp): #evalute new sampling points
            # select the input as arrat
            a_input = np.array(c_subr.pd_sample_record.loc[i,'x1':'x'+str(i_dim)])
            l_output = fun_blackbox(a_input,i_n_rep) #should be i_n_rep-.... and following is the same
            c_subr.pd_sample_record.loc[i,'mean'] = np.mean(l_output)
            c_subr.pd_sample_record.loc[i,'SST'] = sum(turn_to_power(l_output,2))
            c_subr.pd_sample_record.loc[i,'var'] = copy.copy(float(c_subr.pd_sample_record.loc[i,'SST']-i_n_rep*pow(c_subr.pd_sample_record.loc[i,'mean'],2))/(i_n_rep-1))
            c_subr.pd_sample_record.loc[i,'# rep'] = i_n_rep

                               
    # the following update i_min_sample, i_max_sample, f_min_diff_samplemean, and f_max_var
    c_subr.pd_sample_record=c_subr.pd_sample_record.sort_values (by="mean", ascending=True)
    c_subr.pd_sample_record=c_subr.pd_sample_record.reset_index(drop=True) # reindex the sorted df
    if len(c_subr.pd_sample_record)>0:
        c_subr.i_min_sample = c_subr.pd_sample_record.loc[0,'mean']
        c_subr.i_max_sample = c_subr.pd_sample_record.loc[len(c_subr.pd_sample_record)-1,'mean'] 
    c_subr.f_min_diff_samplemean = min(c_subr.pd_sample_record['mean'].shift(-1) - c_subr.pd_sample_record['mean'])
    c_subr.f_max_var = max(c_subr.pd_sample_record.loc[:,'var'])
    #print (c_subr.pd_sample_record)
    return c_subr