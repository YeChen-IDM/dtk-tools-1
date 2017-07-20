# -*- coding: utf-8 -*-
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


def fun_sample_points_generator(l_subr, i_n_sampling, i_n_rep, s_stage):
    import numpy as np
    import pandas as pd

    l_column = ['l_coordinate_lower', 'l_coordinate_upper'] + l_subr[0].l_para + ['Run_Number']
    df_testing_samples = pd.DataFrame([], columns=l_column)  # the dataframe contains the sampling point sent to calibtool
    df_testing_samples['Run_Number'].astype(int)

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
            for i in (i for i in range(0, len(c_subr.pd_sample_record)) if
                      c_subr.pd_sample_record.loc[i, '# rep'] < i_n_rep):  # check enough # reps or not, i is index
                df_testing_samples.append(pd.DataFrame([[c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [c_subr.pd_sample_record.loc[i, p] for p in l_subr[0].l_para] + [i_n_rep - int(c_subr.pd_sample_record.loc[i, '# rep'])]], columns=l_column))

        else:  # if has not enough sampling points and replication
            if len(c_subr.pd_sample_record) >= 1:  # if already has sample points, first deal with them
                for i in (i for i in range(0, len(c_subr.pd_sample_record)) if c_subr.pd_sample_record.loc[i, '# rep'] < i_n_rep):  # check enough # reps or not for existing old sampling points
                    df_testing_samples.append(pd.DataFrame([[c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [c_subr.pd_sample_record.loc[i, p] for p in l_subr[0].l_para] + [i_n_rep - int(c_subr.pd_sample_record.loc[i, '# rep'])]], columns=l_column))

            i_ini_length = len(c_subr.pd_sample_record)  # number of sampling point sofar in this subregion
            for i in range(i_ini_length, i_n_sampling):  # create new rows for new sampling points
                c_subr.pd_sample_record.loc[i] = [1 for n in range(len(c_subr.pd_sample_record.columns))]
                c_subr.pd_sample_record.loc[i, '# rep'] = 0

            for i in range(0, len(l_subr[0].l_para)):  # create new sampling point and add to dataframe
                a_new_sample = np.random.uniform(low=c_subr.l_coordinate_lower[i],
                                                 high=c_subr.l_coordinate_upper[i],
                                                 size=i_n_sampling - i_ini_length)  # generate only for one dim
                index = [x for x in range(i_ini_length, i_n_sampling)]
                c_subr.pd_sample_record.loc[i_ini_length:i_n_sampling - 1, l_subr[0].l_para[i]] = pd.Series(
                    a_new_sample.tolist(), index)
                c_subr.pd_sample_record.loc[index, l_subr[0].l_para[i]] = pd.Series(a_new_sample.tolist(), index)

            for i in range(i_ini_length, i_n_sampling):  # put the new generate sample points in df_samples
                df_testing_samples.append(pd.DataFrame([[c_subr.l_coordinate_lower] + [c_subr.l_coordinate_upper] + [
                    c_subr.pd_sample_record.loc[i, p] for p in l_subr[0].l_para] + [i_n_rep]], columns=l_column))

    return [l_subr, df_testing_samples]
