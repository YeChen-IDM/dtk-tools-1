import math
import numpy as np

from dtk.utils.observations.BaseDistribution import BaseDistribution


class GaussianDistribution(BaseDistribution):
    class InvalidUncertaintyException(Exception): pass

    UNCERTAINTY_CHANNEL = 'two_sigma'

    def prepare(self, dfw, channel, provinciality, age_bins, weight_channel):
        # First verify that the data row uncertainties are set properly (all > 0)
        try:
            uncertainties = dfw._dataframe[self.UNCERTAINTY_CHANNEL]
            n_invalid_uncertainties = uncertainties.where(uncertainties <= 0).count()
        except KeyError:
            n_invalid_uncertainties = len(dfw._dataframe.index)
        if n_invalid_uncertainties > 0:
            raise self.InvalidUncertaintyException('All %s values must be present and positive (>0) for gaussian distributions.' %
                                                   self.UNCERTAINTY_CHANNEL)

        dfw = dfw.filter(keep_only=[channel, self.UNCERTAINTY_CHANNEL, weight_channel])
        self.additional_channels.append(self.UNCERTAINTY_CHANNEL)
        return dfw

    def compare(self, df, reference_channel, data_channel):
        # Note: Might be called extra times by pandas on apply for purposes of "optimization"
        # http://stackoverflow.com/questions/21635915/why-does-pandas-apply-calculate-twice
        #
        log_root_2pi = np.multiply(0.5,np.log(np.multiply(2,np.pi)))

        raw_data = df[reference_channel]
        sim_data = df[data_channel]

        two_sigma = df[self.UNCERTAINTY_CHANNEL]
        if len(two_sigma.unique()) != 1:
            raise Exception('Could not determine what the raw data uncertainty is since reference data varies between replicates.')
        two_sigma = list(two_sigma)[0]

        # ck4, set the default value for uncertainty in the ingest parser

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
