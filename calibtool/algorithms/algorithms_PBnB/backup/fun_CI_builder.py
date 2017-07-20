# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:06:47 2017

@author: TingYu Ho
input:

    l_subergion:examing subregions
    f_delta
    f_epsilon
    f_vol_S:volumn of all regions
    f_vol_C:volumn of set of all undetermined regions
    f_vol_P:volumn of set of all prune regions
    f_vol_M:volumn of set of all maintain regions

outout
    CI_l:lower bound of confident interval
    CI_u:upper bound of condident interval
"""

def fun_CI_builder (l_subr, pd_order_z, f_delta_k, f_alpha_k, f_epsilon):
    from scipy.stats import binom
    import math
    f_vol_S = l_subr[0].f_volumn
    f_vol_C = sum(c.f_volumn for c in l_subr if c.s_label == 'C' and c.b_activate == True) 
    f_vol_P = sum(c.f_volumn for c in l_subr if c.s_label == 'P' and c.b_activate == True) 
    f_vol_M = sum(c.f_volumn for c in l_subr if c.s_label == 'M' and c.b_activate == True)
    f_delta_kl=f_delta_k-float(f_vol_P*f_epsilon)/(f_vol_S*f_vol_C)
    f_delta_ku=f_delta_k+float(f_vol_M*f_epsilon)/(f_vol_S*f_vol_C)
    f_max_r=binom.ppf(f_alpha_k/2, len(pd_order_z), f_delta_kl)
    f_min_s=binom.ppf(1-f_alpha_k/2, len(pd_order_z), f_delta_ku)
    if math.isnan(f_max_r) == True:
        f_max_r = 0
    #print ('f_delta_k: '+str(f_delta_k))
    #print ('f_vol_S: '+str(f_vol_S))
    #print ('f_vol_P: '+str(f_vol_P))
    #print ('f_vol_M: '+str(f_vol_M))
    #print (pd_order_z)
    #print ('max_r: '+str(f_max_r))
    #print ('min_s: '+str(f_min_s))
    CI_l=pd_order_z.loc[f_max_r,'mean']
    CI_u=pd_order_z.loc[f_min_s,'mean']
    
    return [CI_u,CI_l]