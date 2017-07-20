# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:49:42 2017

@author: TingYu Ho
@author: TingYu Ho
This function create the list of worst function used in the step 4
nput: 
    f_CI_l:lower bound confidence interval
    c_subregions:examining subregion
output:
    list 0f update subregions
"""


def fun_elite_indicator(c_subr, f_CI_l):
    if c_subr.i_max_sample < f_CI_l:
        c_subr.b_elite = True
        c_subr.b_worst = False
    return c_subr