import numpy as np
import pandas as pd
import scipy.stats
import math
import copy
from scipy.stats import binom
from c_SubRegion import c_SubRegion
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import os


"""
Created on Fri Jun 23 15:53:08 2017

@author: TingYu Ho
This function uniformly sampling i_n_samp sample points with i_n_rep replication in the subregions c_subregion and generate the df that used to sent to calibtool
input:
    i_n_samp: # sampling needed in the subregion
    i_n_rep: # replication needed in the subregion
    c_subr:examing subregion
outout
    l_subr
    df_testing_samples
"""
# TODO: plotter that can choose any two dimension and fix the value of other dimensions


def fun_sample_points_generator(l_subr, i_n_sampling, i_n_rep, s_stage, l_para):
    l_column = ['l_coordinate_lower', 'l_coordinate_upper'] + l_para + ['replication']
    df_testing_samples = pd.DataFrame([], columns=l_column)  # the dataframe contains the sampling point sent to calibtool
    df_testing_samples['replication'].astype(int)

    if s_stage == 'stage_1':
        l_sampling_subregions = [c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True]
    elif s_stage == 'stage_2':
        l_sampling_subregions = [c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and len(c_subr.pd_sample_record) > 0]
    elif s_stage == 'stage_4_1':
        l_sampling_subregions = [c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and (c_subr.b_worst is True or c_subr.b_elite is True)]
    else:
        l_sampling_subregions = [c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and (c_subr.b_worst is True or c_subr.b_elite is True)]

    for c_subr in l_sampling_subregions:
        c_subr.pd_sample_record = c_subr.pd_sample_record.sort_values(by="mean",
                                                                      ascending=True)  # sort before start
        c_subr.pd_sample_record = c_subr.pd_sample_record.reset_index(drop=True)  # reindex before start
        if len(c_subr.pd_sample_record) >= i_n_sampling:  # if has enough number of sampling points
            #print ('if has enough number of sampling points')
            for i in (i for i in range(0, len(c_subr.pd_sample_record)) if
                      c_subr.pd_sample_record.loc[i, '# rep'] < i_n_rep):  # check enough # reps or not, i is index
                #df_testing_samples.append(pd.DataFrame([[c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [c_subr.pd_sample_record.loc[i, p] for p in l_para] + [i_n_rep - int(c_subr.pd_sample_record.loc[i, '# rep'])]], columns=l_column))
                l_vals = [c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [c_subr.pd_sample_record.loc[i, p] for p in l_para] + [i_n_rep - int(c_subr.pd_sample_record.loc[i, '# rep'])]
                df_testing_samples = df_testing_samples.append(dict(zip(l_column, l_vals)), ignore_index=True)

        else:  # if has not enough sampling points and replication
            #print ('if has not enough sampling points and replication')
            if len(c_subr.pd_sample_record) >= 1:  # if already has sample points, first deal with them
                for i in (i for i in range(0, len(c_subr.pd_sample_record)) if c_subr.pd_sample_record.loc[i, '# rep'] < i_n_rep):  # check enough # reps or not for existing old sampling points
                    #df_testing_samples.append(pd.DataFrame([[c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [c_subr.pd_sample_record.loc[i, p] for p in l_para] + [i_n_rep - int(c_subr.pd_sample_record.loc[i, '# rep'])]], columns=l_column))
                    l_vals = [c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [c_subr.pd_sample_record.loc[i, p] for p in l_para] + [i_n_rep - int(c_subr.pd_sample_record.loc[i, '# rep'])]
                    df_testing_samples = df_testing_samples.append(dict(zip(l_column, l_vals)), ignore_index=True)
            i_ini_length = len(c_subr.pd_sample_record)  # number of sampling point so far in this subregion
            for i in range(i_ini_length, i_n_sampling):  # create new rows for new sampling points
                c_subr.pd_sample_record.loc[i] = [1 for n in range(len(c_subr.pd_sample_record.columns))]
                c_subr.pd_sample_record.loc[i, '# rep'] = 0

            index = [x for x in range(i_ini_length, i_n_sampling)]
            for i in range(0, len(l_para)):  # create new sampling point and add to dataframe
                a_new_sample = np.random.uniform(low=c_subr.l_coordinate_lower[i],
                                                 high=c_subr.l_coordinate_upper[i],
                                                 size=i_n_sampling - i_ini_length)  # generate only for one dim

                c_subr.pd_sample_record.loc[i_ini_length:i_n_sampling - 1, l_para[i]] = pd.Series(
                    a_new_sample.tolist(), index)
                c_subr.pd_sample_record.loc[index, l_para[i]] = pd.Series(a_new_sample.tolist(), index)
            c_subr.pd_sample_record.loc[index, 'mean'] = 0
            c_subr.pd_sample_record.loc[index, 'var'] = 0
            c_subr.pd_sample_record.loc[index, 'SST'] = 0
            for i in range(i_ini_length, i_n_sampling):  # put the new generate sample points in df_samples
                l_vals = [c_subr.l_coordinate_lower]+[c_subr.l_coordinate_upper]+[c_subr.pd_sample_record.loc[i, p] for p in l_para] + [i_n_rep]
                df_testing_samples = df_testing_samples.append(dict(zip(l_column, l_vals)), ignore_index=True)
    return [l_subr, df_testing_samples]


def turn_to_power(list, power):
    return [number**power for number in list]


def fun_results_organizer(l_subr, df_testing_samples, params):
    print ('=====fun_results_organizer=====')
    # df_testing_samples: ['l_coordinate_lower', 'l_coordinate_upper'] + l_para + ['replication'] + ['result']
    # result contains the list of outputs
    df_testing_samples = df_testing_samples.reset_index(drop=True)
    for i in range(0, len(df_testing_samples)):
        for c_subr in [c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and c_subr.l_coordinate_lower == df_testing_samples.loc[i, 'l_coordinate_lower'] and c_subr.l_coordinate_upper == df_testing_samples.loc[i, 'l_coordinate_upper']]:
            c_subr.pd_sample_record = c_subr.pd_sample_record.reset_index(drop=True)  # reindex the sorted df

            for j in range(0, len(c_subr.pd_sample_record)):
                if ([c_subr.pd_sample_record.loc[j, p['Name']] for p in params] == [df_testing_samples.loc[i, p['Name']]
                                                                                    for p in params]):
                    i_n_rep = (c_subr.pd_sample_record.loc[j, '# rep'] + len(df_testing_samples.loc[i, 'result']))
                    if c_subr.pd_sample_record.loc[j, '# rep'] == 0:
                        c_subr.pd_sample_record.loc[j, 'mean'] = np.mean(df_testing_samples.loc[i, 'result'])
                        c_subr.pd_sample_record.loc[j, 'SST'] = sum(turn_to_power(df_testing_samples.loc[i, 'result'], 2))
                        c_subr.pd_sample_record.loc[j, 'var'] = np.var(df_testing_samples.loc[i, 'result'])
                        c_subr.pd_sample_record.loc[j, '# rep'] = i_n_rep
                    else:
                        c_subr.pd_sample_record.loc[j, 'mean'] = copy.copy(float(
                            int(c_subr.pd_sample_record.loc[i, '# rep']) * c_subr.pd_sample_record.loc[i, 'mean'] + sum(df_testing_samples.loc[i, 'result'])) / i_n_rep)
                        c_subr.pd_sample_record.loc[j, 'SST'] = copy.copy(c_subr.pd_sample_record.loc[i, 'SST'] + sum(turn_to_power(df_testing_samples.loc[i, 'result'], 2)))
                        c_subr.pd_sample_record.loc[j, 'var'] = copy.copy(float(c_subr.pd_sample_record.loc[i, 'SST'] - i_n_rep * pow(c_subr.pd_sample_record.loc[i, 'mean'], 2)) / (i_n_rep - 1))
                        c_subr.pd_sample_record.loc[j, '# rep'] = i_n_rep

        # the following update i_min_sample, i_max_sample, f_min_diff_sample_mean, and f_max_var
        c_subr.pd_sample_record = c_subr.pd_sample_record.sort_values(by="mean", ascending=True)
        c_subr.pd_sample_record = c_subr.pd_sample_record.reset_index(drop=True)  # reindex the sorted df
        if len(c_subr.pd_sample_record) > 0:
            c_subr.i_min_sample = c_subr.pd_sample_record.loc[0, 'mean']
            c_subr.i_max_sample = c_subr.pd_sample_record.loc[len(c_subr.pd_sample_record) - 1, 'mean']
        c_subr.f_min_diff_sample_mean = min(
            c_subr.pd_sample_record['mean'].shift(-1) - c_subr.pd_sample_record['mean'])
        c_subr.f_max_var = max(c_subr.pd_sample_record.loc[:, 'var'])
    return l_subr

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
    c_subr.pd_sample_record = c_subr.pd_sample_record.sort_values(by="mean", ascending=True)
    c_subr.pd_sample_record = c_subr.pd_sample_record.reset_index(drop=True)  # reindex the sorted df
    if len(c_subr.pd_sample_record) > 0:
        c_subr.i_min_sample = c_subr.pd_sample_record.loc[0, 'mean']
        c_subr.i_max_sample = c_subr.pd_sample_record.loc[len(c_subr.pd_sample_record)-1, 'mean']
    c_subr.f_min_diff_sample_mean = min(c_subr.pd_sample_record['mean'].shift(-1) - c_subr.pd_sample_record['mean'])
    c_subr.f_max_var = max(c_subr.pd_sample_record.loc[:, 'var'])
    return c_subr

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
    if list(i.f_min_diff_sample_mean for i in l_subr if i.s_label == 'C' and i.b_activate is True) + [] == []: # to prevent empty sequence
        f_d_star = 0.005
    elif min(i.f_min_diff_sample_mean for i in l_subr if i.s_label == 'C' and i.b_activate is True) < 0.005:
        f_d_star = 0.005
    else:
        f_d_star = min(i.f_min_diff_sample_mean for i in l_subr if i.s_label == 'C' and i.b_activate is True)
    f_var_star = max(i.f_max_var for i in l_subr if i.s_label == 'C' and i.b_activate is True)
    z = scipy.stats.norm.ppf(1 - f_alpha / 2)
    # to prevent the float NaN
    if math.isnan(z) is True or math.isnan(f_d_star) is True or math.isnan(f_var_star) is True:
        i_n_rep = i_n_rep
    else:
        i_n_rep = max(i_n_rep, 4 * int(math.ceil(pow(z, 2) * f_var_star / pow(f_d_star, 2))))
    return i_n_rep

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
    l_all_sample_C = []
    for i in (i for i in l_subr if i.s_label == 'C'):
        l_all_sample_C.append(i.pd_sample_record)
    pd_order_z = pd.concat(l_all_sample_C)
    pd_order_z = pd_order_z.sort_values(by="mean", ascending=True)
    pd_order_z = pd_order_z.reset_index(drop=True)  # reindex the sorted df
    return pd_order_z

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


def fun_pruning_indicator(l_subr, f_CI_u):
    for c_subr in (c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and c_subr.b_worst is True):
        if c_subr.i_min_sample > f_CI_u:
            c_subr.b_maintaining_indicator = False
            c_subr.b_pruning_indicator = True
    return l_subr

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


def fun_maintaining_indicator(l_subr, f_CI_l):
    for c_subr in (c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and c_subr.b_elite is True):
        if c_subr.i_max_sample < f_CI_l:
            c_subr.b_maintaining_indicator = True
            c_subr.b_pruning_indicator = False
    return l_subr

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


def fun_elite_indicator(l_subr, f_CI_l):
    for c_subr in (c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and c_subr.i_max_sample < f_CI_l):
        c_subr.b_elite = True
        c_subr.b_worst = False
    return l_subr

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


def fun_worst_indicator(l_subr, f_CI_u):
    for c_subr in (c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True and c_subr.i_min_sample > f_CI_u):
        c_subr.b_elite = False
        c_subr.b_worst = True
    return l_subr

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


def fun_quantile_update(l_subr, f_delta):
    f_vol_C = sum(c.f_volume for c in l_subr if c.s_label == 'C' and c.b_activate is True)
    f_vol_pruning = sum(c.f_volume for c in l_subr if c.b_pruning_indicator is True and c.b_activate is True)
    f_vol_maintaining = sum(c.f_volume for c in l_subr if c.b_maintaining_indicator is True and c.b_activate is True)
    f_delta = float(f_delta*f_vol_C-f_vol_maintaining)/(f_vol_C-f_vol_pruning-f_vol_maintaining)
    return f_delta


"""
Created on Fri Jun 23 16:06:47 2017

@author: TingYu Ho
input:

    l_subergion:examine subregions
    f_delta
    f_epsilon
    f_vol_S:volume of all regions
    f_vol_C:volume of set of all undetermined regions
    f_vol_P:volume of set of all prune regions
    f_vol_M:volume of set of all maintain regions

outout
    CI_l:lower bound of confident interval
    CI_u:upper bound of confident interval
"""


def fun_CI_builder(l_subr, pd_order_z, f_delta_k, f_alpha_k, f_epsilon):
    f_vol_S = l_subr[0].f_volume
    f_vol_C = sum(c.f_volume for c in l_subr if c.s_label == 'C' and c.b_activate is True)
    f_vol_P = sum(c.f_volume for c in l_subr if c.s_label == 'P' and c.b_activate is True)
    f_vol_M = sum(c.f_volume for c in l_subr if c.s_label == 'M' and c.b_activate is True)
    f_delta_kl = f_delta_k - float(f_vol_P * f_epsilon) / (f_vol_S * f_vol_C)
    f_delta_ku = f_delta_k + float(f_vol_M * f_epsilon) / (f_vol_S * f_vol_C)
    f_max_r = binom.ppf(f_alpha_k / 2, len(pd_order_z), f_delta_kl)
    f_min_s = binom.ppf(1 - f_alpha_k / 2, len(pd_order_z), f_delta_ku)
    if math.isnan(f_max_r) is True:
        f_max_r = 0
    # print ('f_delta_k: '+str(f_delta_k))
    # print ('f_vol_S: '+str(f_vol_S))
    # print ('f_vol_P: '+str(f_vol_P))
    # print ('f_vol_M: '+str(f_vol_M))
    # print (pd_order_z)
    # print ('max_r: '+str(f_max_r))
    # print ('min_s: '+str(f_min_s))
    CI_l = pd_order_z.loc[f_max_r, 'mean']
    CI_u = pd_order_z.loc[f_min_s, 'mean']
    return [CI_u, CI_l]

"""
Created on Thu Jun 29 10:45:17 2017

@author: TingYu
"""


def fun_pruning_labeler(l_subr):
    for c_subr in (c_subr for c_subr in l_subr if c_subr.b_pruning_indicator is True and c_subr.b_activate is True):  # <-- whose  worst == 1
        c_subr.s_label = 'P'
    return l_subr

"""
Created on Thu Jun 29 10:46:09 2017

@author: TingYu
"""


def fun_maintaining_labeler(l_subr):
    for c_subr in (c_subr for c_subr in l_subr if c_subr.b_maintaining_indicator is True and c_subr.b_activate is True):  # <-- whose elite == 1
        c_subr.s_label = 'M'
    return l_subr

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


def fun_reg_branching(c_subr, i_n_branching, params, s_branching_dim):
    i_max_index = [p['Name'] for p in params].index(s_branching_dim)

    l_subr_new = []
    # the following creates B subregions in the list of subregions
    for i in range(0, i_n_branching):
        l_coordinate_lower = copy.deepcopy(c_subr.l_coordinate_lower)
        l_coordinate_upper = copy.deepcopy(c_subr.l_coordinate_upper)
        l_coordinate_lower[i_max_index] = float((c_subr.l_coordinate_upper[i_max_index] - c_subr.l_coordinate_lower[i_max_index])*i)/i_n_branching+c_subr.l_coordinate_lower[i_max_index]
        l_coordinate_upper[i_max_index] = float((c_subr.l_coordinate_upper[i_max_index] - c_subr.l_coordinate_lower[i_max_index])*(i+1))/i_n_branching+c_subr.l_coordinate_lower[i_max_index]
        l_new_branching_subr = c_SubRegion(l_coordinate_lower, l_coordinate_upper, params)
        l_subr_new.append(l_new_branching_subr)
    # the following reallocate the sampling points
    for i in l_subr_new:
        i.pd_sample_record = c_subr.pd_sample_record[(c_subr.pd_sample_record[s_branching_dim] > i.l_coordinate_lower[i_max_index]) & (c_subr.pd_sample_record[s_branching_dim] < i.l_coordinate_upper[i_max_index])]
    for i in l_subr_new:  # reindex the sampling points into 0 1 2...
        i.pd_sample_record = i.pd_sample_record.reset_index(drop=True)
        # update attributed based on data
        if len(i.pd_sample_record) > 0:
            i.i_min_sample = min(i.pd_sample_record.loc[:, 'mean'])
            i.i_max_sample = max(i.pd_sample_record.loc[:, 'mean'])
            i.f_min_diff_sample_mean = min(i.pd_sample_record['mean'].shift(-1) - i.pd_sample_record['mean'])
        if len(i.pd_sample_record) > 1:
            i.f_max_var = max(i.pd_sample_record.loc[:, 'var'])
    return l_subr_new


def fun_plot2D(l_subr, l_ini_coordinate_lower, l_ini_coordinate_upper, params, str_k):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(l_ini_coordinate_upper[0], l_ini_coordinate_upper[1])
    ax.plot(l_ini_coordinate_lower[0], l_ini_coordinate_lower[1])

    for i in (i for i in l_subr if i.b_activate is True):
        if i.s_label == 'M':
            alpha_value = 1
        elif i.s_label == 'P':
            alpha_value = 0.1
        else:  # i.s_label == 'C':
            alpha_value = 0.6
        ax.add_patch(
            patches.Rectangle(
                (i.l_coordinate_lower[0], i.l_coordinate_lower[1]),  # (x,y)
                i.l_coordinate_upper[0] - i.l_coordinate_lower[0],  # width
                i.l_coordinate_upper[1] - i.l_coordinate_lower[1],  # height
                alpha=alpha_value,
                edgecolor="black"
            )
        )
    # the following plot the minimum and maximum point
    df_all_sample = pd.concat([c_subr.pd_sample_record for c_subr in l_subr if c_subr.b_activate is True])
    df_all_sample = df_all_sample.sort_values(by="mean", ascending=True)  # sort before start
    df_all_sample = df_all_sample.reset_index(drop=True)
    f_min_value = df_all_sample.loc[0, 'mean']
    f_max_value = df_all_sample.loc[len(df_all_sample) - 1, 'mean']
    l_min_coordinate = [df_all_sample.loc[0, p['Name']] for p in params]
    l_max_coordinate = [df_all_sample.loc[len(df_all_sample) - 1, p['Name']] for p in params]
    #red_patch = patches.Patch(color='red', marker='x', label='The minimum value:'+str(f_min_value))
    #blue_patch = patches.Patch(color='blue', marker='o', label='The maximum value:'+str(f_max_value))
    p_min, p_max = ax.plot(l_min_coordinate[0], l_min_coordinate[1], '*b', l_max_coordinate[0], l_max_coordinate[1], 'or')
    fig.legend((p_min, p_max), ('minimum point:['+str(l_min_coordinate[0])+','+str(l_min_coordinate[1])+'], result:'+str(f_min_value), 'maximum point:['+str(l_max_coordinate[0])+','+str(l_max_coordinate[1])+'], result:'+str(f_max_value)), 'upper right')

    # plt.legend(handles=[red_patch, blue_patch])

    ax.set_xlabel([p['Name'] for p in params][0])
    ax.set_ylabel([p['Name'] for p in params][1])

    fig.savefig('test_iteration '+str_k+'.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    with open('test_l_subr_iteration'+str_k+'.dat', "wb") as f:
        pickle.dump(l_subr, f)

    plt.close(fig)
