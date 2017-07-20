# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:19:38 2017

@author: TingYu Ho
This function orders all the samplign points in all the undetermined regions
input:
    l_subr:list of all subregions
output:
    pd_order_z:dataframe of descending order data
"""

def fun_order_region(l_subr):
    import pandas as pd 
    
    l_allsample_C=[]
    for i in (i for i in l_subr if i.s_label=='C'): 
        l_allsample_C.append(i.pd_sample_record)
    pd_order_z = pd.concat(l_allsample_C)     
    pd_order_z=  pd_order_z.sort_values (by="mean", ascending=True)
    pd_order_z = pd_order_z.reset_index(drop=True) # reindex the sorted df
    return pd_order_z