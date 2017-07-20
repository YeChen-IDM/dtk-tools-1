# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:57:51 2017

@author: TingYu Ho
This function update the quantile
input:
    l_subr: list of subregions
    f_delta
output:
    f_delta:updated quantile
"""

def fun_quantile_update (l_subr, f_delta):
    f_vol_C = sum(c.f_volumn for c in l_subr if c.s_label == 'C' and c.b_activate == True) 
    f_vol_Pruning = sum(c.f_volumn for c in l_subr if c.b_pruning_indicator == True and c.b_activate == True) 
    f_vol_Maintaining = sum(c.f_volumn for c in l_subr if c.b_maintaining_indicator == True and c.b_activate == True)
    #print (f_vol_C)
    #print (f_vol_Pruning)
    #print (f_vol_Maintaining)
    #print (f_delta)
    f_delta = float(f_delta*f_vol_C-f_vol_Maintaining)/(f_vol_C-f_vol_Pruning-f_vol_Maintaining)
    #print (f_delta)
    return f_delta