# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:49:42 2017

@author: TingYu Ho
This function create the list of worst function used in the step 4
nput: 
    f_CI_u:upper bound confidence interval
    c_subregions:examining subregion
output:
    list 0f updated subregions
"""

def fun_worst_indicator (c_subr, f_CI_u):
    if c_subr.i_min_sample > f_CI_u:
        c_subr.b_elite=False
        c_subr.b_worst=True
    return c_subr