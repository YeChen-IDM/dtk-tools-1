# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:49:41 2017

@author: TingYu Ho
This function create the list of subregions prepared to maintain from the list of elite subregions 
nput: 
    f_CI_l:lower bound confidence interval
    c_subr: examining subregion
output:
   update subregions
"""

def fun_maintaining_indicator(c_subr, f_CI_l):
    if c_subr.i_max_sample < f_CI_l:
        c_subr.b_maintaining_indicator = True
        c_subr.b_pruning_indicator = False
    return c_subr