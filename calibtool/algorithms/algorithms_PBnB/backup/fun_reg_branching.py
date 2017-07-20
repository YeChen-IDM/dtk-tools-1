# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:07:27 2017

@author: TingYu Ho
input:
    c_subr: examining subregion
output:
    l_subr: list of branching B subregions
How do I copy an object in Python?
http://effbot.org/pyfaq/how-do-i-copy-an-object-in-python.htm
"""

def fun_reg_branching(c_subr, i_dim, i_B):
    import numpy as np
    import copy
    # the following decides which diminesion should be divide
    my_list = (np.array(c_subr.l_coordinate_upper)-np.array(c_subr.l_coordinate_lower)).tolist()
    f_max_value = max(my_list)
    i_max_index = my_list.index(f_max_value)
    s_branching_dim = 'x'+str(i_max_index+1) # index
    l_subr = []  
    # the following creates B subregions in the list of subregions     
    for i in range(0,i_B):
        l_coordinate_lower = copy.deepcopy(c_subr.l_coordinate_lower)
        l_coordinate_upper = copy.deepcopy(c_subr.l_coordinate_upper)
        l_coordinate_lower[i_max_index] = float((c_subr.l_coordinate_upper[i_max_index]- c_subr.l_coordinate_lower[i_max_index])*i)/i_B+c_subr.l_coordinate_lower[i_max_index]
        l_coordinate_upper[i_max_index] = float((c_subr.l_coordinate_upper[i_max_index]- c_subr.l_coordinate_lower[i_max_index])*(i+1))/i_B+c_subr.l_coordinate_lower[i_max_index]
        l_new_branching_subr = c_SubRegion(i_dim, l_coordinate_lower, l_coordinate_upper)
        l_subr.append(l_new_branching_subr)
    # the following reallocate sthe sampling points
    for i in l_subr:
        i.pd_sample_record = c_subr.pd_sample_record[(c_subr.pd_sample_record[s_branching_dim] > i.l_coordinate_lower[i_max_index]) & (c_subr.pd_sample_record[s_branching_dim] < i.l_coordinate_upper[i_max_index]) ]
    for i in l_subr: # reindex the sampling points into 0 1 2...
        i.pd_sample_record = i.pd_sample_record.reset_index(drop=True)
        # update attributed based on datas
        if len(i.pd_sample_record) > 0:
            i.i_min_sample = min(i.pd_sample_record.loc[:,'mean'])
            i.i_max_sample = max(i.pd_sample_record.loc[:,'mean']) 
            i.f_min_diff_samplemean = min(i.pd_sample_record['mean'].shift(-1) - i.pd_sample_record['mean'])
        if len(i.pd_sample_record) > 1:
            i.f_max_var = max(i.pd_sample_record.loc[:,'var'])
    return l_subr