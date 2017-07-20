# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:19:38 2017

@author: TingYu Ho
This function orders all the samplign points in all the undetermined regions
input:
    one subregions
output:
    c_subregion:subregion with updated dataframe of descending order data
"""

def fun_order_subregion(c_subr):
    c_subr.pd_sample_record=c_subr.pd_sample_record.sort_values (by="mean", ascending=True)
    c_subr.pd_sample_record=c_subr.pd_sample_record.reset_index(drop=True) # reindex the sorted df
    if len(c_subr.pd_sample_record)>0:
        c_subr.i_min_sample = c_subr.pd_sample_record.loc[0,'mean']
        c_subr.i_max_sample = c_subr.pd_sample_record.loc[len(c_subr.pd_sample_record)-1,'mean'] 
    c_subr.f_min_diff_samplemean = min(c_subr.pd_sample_record['mean'].shift(-1) - c_subr.pd_sample_record['mean'])
    c_subr.f_max_var = max(c_subr.pd_sample_record.loc[:,'var'])
    return c_subr