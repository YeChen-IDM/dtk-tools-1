import numpy as np
from scipy.special import gammaln
from scipy.stats import beta

from dtk.utils.observations.BaseDistribution import BaseDistribution
from dtk.utils.observations.PopulationObs import PopulationObs


class BetaDistribution(BaseDistribution):

    def prepare(self, dfw, channel, provinciality, age_bins):
        self.alpha_channel, self.beta_channel = dfw.add_beta_parameters(channel=channel,
                                                                        provinciality=provinciality,
                                                                        age_bins=age_bins)
        dfw = dfw.filter(keep_only=[channel, self.alpha_channel, self.beta_channel, PopulationObs.WEIGHT_CHANNEL])
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
