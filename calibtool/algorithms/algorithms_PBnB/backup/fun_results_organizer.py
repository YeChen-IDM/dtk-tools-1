

def fun_results_organizer(l_subr, df_testing_samples):
    import numpy as np
    import copy
    from scipy.ndimage import variance


    def turn_to_power(list, power):
        return [number**power for number in list]

    #  df_testing_samples: ['l_coordinate_lower', 'l_coordinate_upper'] + l_subr[0].l_para + ['Run_Number'] + ['result']
    for c_subr in [c_subr for c_subr in l_subr if c_subr.s_label == 'C' and c_subr.b_activate is True]:
        i_n_rep = (c_subr.pd_sample_record.loc[i, '# rep']+len(df_testing_samples.loc[i, 'result']))
        for i in range(0, len(df_testing_samples)):
            if c_subr.l_coordinate_lower == df_testing_samples.loc[i, 'l_coordinate_lower'] and \
                            c_subr.l_coordinate_upper == df_testing_samples.loc[i, 'l_coordinate_upper']:
                if c_subr.pd_sample_record.loc[i, '# rep'] == 0:
                    c_subr.pd_sample_record.loc[i, 'mean'] = np.mean(df_testing_samples.loc[i, 'result'])
                    c_subr.pd_sample_record.loc[i, 'SST'] = sum(turn_to_power(df_testing_samples.loc[i, 'result'], 2))
                    c_subr.pd_sample_record.loc[i, 'var'] = variance(df_testing_samples.loc[i, 'result'])
                    c_subr.pd_sample_record.loc[i, '# rep'] = i_n_rep
                else:
                    c_subr.pd_sample_record.loc[i, 'mean'] = copy.copy(float(
                        int(c_subr.pd_sample_record.loc[i, '# rep']) * c_subr.pd_sample_record.loc[i, 'mean'] + sum(df_testing_samples.loc[i, 'result'])) / i_n_rep)
                    c_subr.pd_sample_record.loc[i, 'SST'] = copy.copy(c_subr.pd_sample_record.loc[i, 'SST'] + sum(turn_to_power(df_testing_samples.loc[i, 'result'], 2)))
                    c_subr.pd_sample_record.loc[i, 'var'] = copy.copy(float(c_subr.pd_sample_record.loc[i, 'SST'] - i_n_rep * pow(c_subr.pd_sample_record.loc[i, 'mean'], 2)) / (i_n_rep - 1))
                    c_subr.pd_sample_record.loc[i, '# rep'] = i_n_rep

        # the following update i_min_sample, i_max_sample, f_min_diff_samplemean, and f_max_var
        c_subr.pd_sample_record = c_subr.pd_sample_record.sort_values(by="mean", ascending=True)
        c_subr.pd_sample_record = c_subr.pd_sample_record.reset_index(drop=True)  # reindex the sorted df
        if len(c_subr.pd_sample_record) > 0:
            c_subr.i_min_sample = c_subr.pd_sample_record.loc[0, 'mean']
            c_subr.i_max_sample = c_subr.pd_sample_record.loc[len(c_subr.pd_sample_record) - 1, 'mean']
        c_subr.f_min_diff_samplemean = min(
            c_subr.pd_sample_record['mean'].shift(-1) - c_subr.pd_sample_record['mean'])
        c_subr.f_max_var = max(c_subr.pd_sample_record.loc[:, 'var'])
    return l_subr
