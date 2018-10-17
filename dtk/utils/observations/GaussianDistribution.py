import math
import numpy as np

from dtk.utils.observations.BaseDistribution import BaseDistribution


class GaussianDistribution(BaseDistribution):

    def compare(self, df, reference_channel, data_channel):
        # Note: Might be called extra times by pandas on apply for purposes of "optimization"
        # http://stackoverflow.com/questions/21635915/why-does-pandas-apply-calculate-twice
        #
        log_root_2pi = np.multiply(0.5,np.log(np.multiply(2,np.pi)))

        raw_data = df[reference_channel]
        sim_data = df[data_channel]

        # set the reference data uncertainty and verify that each replicate has the same reference value
        raw_data_mean = raw_data.mean()
        if raw_data_mean == list(raw_data)[0]:
            two_sigma = raw_data_mean / 5
        else:
            raise Exception('Could not determine what the raw data uncertainty is since reference data varies between replicates.')

        raw_data_variance = np.divide(two_sigma, 2)**2

        # return np.subtract(raw_data,sim_data)

        log_of_gaussian = - log_root_2pi - np.multiply(0.5, np.log(raw_data_variance)) -\
                          np.divide(np.multiply(0.5, ((sim_data - raw_data)**2)), raw_data_variance)

        # add likelihood columns to df
        df_sample_PA = df.copy()

        df_sample_PA['log_of_gaussian'] = log_of_gaussian

        largest_possible_log_of_gaussian = 0
        largest_possible_log_of_gaussian = largest_possible_log_of_gaussian + (np.multiply(-1, log_root_2pi) - np.multiply(0.5, math.log(raw_data_variance)))

        df_sample_PA['lplg'] = largest_possible_log_of_gaussian
        df_sample_PA['scale_min'] = -708.3964
        df_sample_PA['scale_max'] = 100

        conditions = [
            df_sample_PA['log_of_gaussian'] <= df_sample_PA['scale_min'],
            df_sample_PA['log_of_gaussian'] > df_sample_PA['scale_min']]

        choices = [df_sample_PA['scale_min'], df_sample_PA['log_of_gaussian']+df_sample_PA['scale_max']-df_sample_PA['lplg']]

        df_sample_PA['scaled_log_of_gaussian'] = np.select(conditions, choices, default=-708.3964)

        df_sample_PA['mean_of_gaussian'] = df_sample_PA['scaled_log_of_gaussian'].mean()

        return df_sample_PA['scaled_log_of_gaussian'].mean()
