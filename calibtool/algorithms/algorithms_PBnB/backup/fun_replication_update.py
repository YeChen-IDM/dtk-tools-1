# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:41:44 2017

@author: TingYu Ho
f_update_replication function is aim to calculate the updated replication number
input:
    l_subr:list of all examing subregions
    i_n_rep:original replication
    f_alpha
outout
    i_replication:update replication
note: for more information about "Get the object with the max attribute's value in a list of objects"
https://stackoverflow.com/questions/18005172/get-the-object-with-the-max-attributes-value-in-a-list-of-objects

Jun 29 2017
	assign an minimun value for f_D_star to avoid huge i_n_rep output
"""

def fun_replication_update(l_subr, i_n_rep, f_alpha):
    import scipy.stats
    import math
    if list(i.f_min_diff_samplemean for i in l_subr if i.s_label == 'C' and i.b_activate == True)+[] == []: # to prevent empty sequence
        f_d_star = 0.005
    elif min(i.f_min_diff_samplemean for i in l_subr if i.s_label == 'C' and i.b_activate == True) < 0.005:
        f_d_star = 0.005
    else:
        f_d_star = min(i.f_min_diff_samplemean for i in l_subr if i.s_label == 'C' and i.b_activate == True)
    #print ('f_d_star: '+str(f_d_star))
    for i in (i for i in l_subr if i.b_activate == True and i.s_label == 'C'):
        print (i.f_max_var)
    f_var_star = max(i.f_max_var  for i in l_subr if i.s_label == 'C' and i.b_activate == True)
    #print ('f_var_star: '+str(f_var_star))
    z=scipy.stats.norm.ppf(1-f_alpha/2)
    #print ('z: '+str(z))
    #print ('f_d_star: '+str(f_d_star))
    #print ('f_var_star: '+str(f_var_star))
    # to prevent the float NaN
    if math.isnan(z) == True or math.isnan(f_d_star) == True or math.isnan(f_var_star) == True:
        i_n_rep = i_n_rep
    else: 
        i_n_rep = max(i_n_rep,4*int(math.ceil(pow(z,2)*f_var_star/pow(f_d_star,2))))
    return i_n_rep

    