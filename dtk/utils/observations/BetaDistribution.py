import numpy as np
from scipy.special import gammaln
from scipy.stats import beta

from dtk.utils.observations.BaseDistribution import BaseDistribution


class BetaDistribution(BaseDistribution):
    class InvalidEffectiveCount(Exception): pass

    COUNT_CHANNEL = 'effective_count'

    def prepare(self, dfw, channel, provinciality, age_bins, weight_channel):
        # First verify that the data row counts are set properly (all > 0)
        try:
            counts = dfw._dataframe[self.COUNT_CHANNEL]
            n_invalid_counts = counts.where(counts <= 0).count()
        except KeyError:
            n_invalid_counts = len(dfw._dataframe.index)
        if n_invalid_counts > 0:
            raise self.InvalidEffectiveCount('All %s values must be present and positive (>0) for beta distributions.' %
                                             self.COUNT_CHANNEL)

        # filter before adding beta params to make sure to not alter the input dfw parameter object
        dfw = dfw.filter(keep_only=[channel, self.COUNT_CHANNEL, weight_channel])
        self.alpha_channel, self.beta_channel = dfw.add_beta_parameters(channel=channel,
                                                                        provinciality=provinciality,
                                                                        age_bins=age_bins)
        for ch in [self.alpha_channel, self.beta_channel]:
            self.additional_channels.append(ch)
        return dfw

    def compare(self, df, reference_channel, data_channel):
        a = df[self.alpha_channel]
        b = df[self.beta_channel]
        x = df[data_channel]

        # This is what we're calculating:
        # BETA(output_i | alpha=alpha(Data), beta = beta(Data) )
        betaln = np.multiply((a - 1), np.log(x)) \
                 + np.multiply((b - 1), np.log(1 - x)) \
                 - (gammaln(a) + gammaln(b) - gammaln(a + b))

        # Replace -inf with log(machine tiny)
        betaln[np.isinf(betaln)] = self.LOG_FLOAT_TINY

        df_sample = df.copy()
        df_sample['betaln'] = betaln

        x_mode = np.divide((a - 1), (a + b - 2))
        largest_possible_log_of_beta = beta.logpdf(x_mode, a, b)

        lob = beta.logpdf(x, a, b)

        df_sample['lplb'] = largest_possible_log_of_beta
        df_sample['lob'] = lob
        df_sample['scale_min'] = -708.3964
        df_sample['scale_max'] = 100

        conditions = [
            df_sample['betaln'] <= df_sample['scale_min'],
            df_sample['betaln'] > df_sample['scale_min']]

        choices = [df_sample['scale_min'], df_sample['lob'] + df_sample['scale_max'] - df_sample['lplb']]

        df_sample['scaled_betaln'] = np.select(conditions, choices, default=-708.3964)

        df_sample['mean_of_betaln'] = df_sample['scaled_betaln'].mean()

        return df_sample['mean_of_betaln'].mean()
