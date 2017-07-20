# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:46:06 2017

@author: TingYu Ho
This function create the list of subregions prepared to prune from the list of worst subregions 
nput: 
    f_CI_u:upper bound confidence interval
     c_subr: examining subregion
output:
    update subregions
"""

def fun_pruning_indicator (c_subr, f_CI_u):
    if c_subr.i_min_sample > f_CI_u:
        c_subr.b_maintaining_indicator = False
        c_subr.b_pruning_indicator = True
    return c_subr