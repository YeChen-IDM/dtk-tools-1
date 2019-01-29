import numpy as np
import pandas as pd

from scipy.special import gammaln
from scipy.stats import beta

from dtk.utils.observations.BaseDistribution import BaseDistribution


class BetaDistribution(BaseDistribution):
    class InvalidEffectiveCountException(Exception): pass
    class InvalidCountChannelException(Exception): pass

    COUNT_CHANNEL = 'effective_count'
    UNCERTAINTY_CHANNEL = COUNT_CHANNEL  # ugly, but prevents code refactor of things using COUNT_CHANNEL. And
    # keeps THat code more readable.

    def prepare(self, dfw, channel, weight_channel=None, additional_keep=None):
        additional_keep = additional_keep or []

        # help people correct their old habits; 'Count' has been replaced with 'effective_count'
        if 'Count' in (dfw.stratifiers + dfw.channels):
            raise self.InvalidCountChannelException('Count is no longer used as a channel. '
                                             'Add %s to the additional columns list in your ingest form instead.' %
                                             self.COUNT_CHANNEL)

        # First verify that the data row counts are set properly (all > 0)
        try:
            counts = dfw._dataframe[self.COUNT_CHANNEL]
            n_invalid_counts = counts.where(counts <= 0).count()
        except KeyError:
            n_invalid_counts = len(dfw._dataframe.index)
        if n_invalid_counts > 0:
            raise self.InvalidEffectiveCountException('All %s values must be present and positive (>0) for beta distributions.' %
                                                      self.COUNT_CHANNEL)

        # filter before adding beta params to make sure to not alter the input dfw parameter object
        channels_to_keep = [channel, self.COUNT_CHANNEL]+additional_keep
        channels_to_keep = channels_to_keep + [weight_channel] if weight_channel is not None else channels_to_keep
        dfw = dfw.filter(keep_only=channels_to_keep)
        self.alpha_channel, self.beta_channel = self.add_beta_parameters(dfw=dfw, channel=channel)
        self.additional_channels += [self.alpha_channel, self.beta_channel]
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

    @staticmethod
    def construct_beta_channel(channel, type):
        # age_bins = age_bins if isinstance(age_bins, list) else [age_bins]
        # age_bin_str = '_'.join([str(age_bin) for age_bin in age_bins])
        return '%s--Beta-%s' % (channel, type)

    def add_percentile_values(self, dfw, channel, p):
        from scipy.stats import beta

        alpha_channel = self.construct_beta_channel(channel=channel, type='alpha')
        beta_channel = self.construct_beta_channel(channel=channel, type='beta')
        required_items = [alpha_channel, beta_channel]
        try:
            dfw.verify_required_items(needed=required_items)
        except dfw.MissingRequiredData:
            self.add_beta_parameters(dfw=dfw, channel=channel)

        values = beta.ppf(p, dfw._dataframe[alpha_channel], dfw._dataframe[beta_channel])
        p_channel = self.construct_beta_channel(channel=channel, type=p)
        values_df = pd.DataFrame({p_channel: values})
        dfw._dataframe = dfw._dataframe.join(values_df)
        new_channels = [p_channel]
        return new_channels

    def add_beta_parameters(self, dfw, channel):
        """
        Compute and add alpha, beta parameters for a beta distribution to the current self._dataframe object.
            Distribution is computed for the provided channel (data field), using 'count'. Result is put into new
            channels/columns named <channel>--Beta-alpha, <channel>--Beta-beta. If both alpha/beta channels already
            exist in the dataframe, nothing is computed.
        :param channel: The data channel/column to compute the beta distribution for.
        :return: a list of the channel-associated alpha and beta parameter channel names.
        """
        required_data = [self.COUNT_CHANNEL, channel]
        dfw.verify_required_items(needed=required_data)

        alpha_channel = self.construct_beta_channel(channel=channel, type='alpha')
        beta_channel = self.construct_beta_channel(channel=channel, type='beta')
        new_channels = [alpha_channel, beta_channel]

        # Useful for an 'omg what is going on!' type of check
        for ch in new_channels:
            if ch in dfw._dataframe.columns:
                raise Exception('Channel %s already exists in dataframe.' % ch)

        # only add the alpha/beta channels if they are not already present
        if alpha_channel not in dfw.channels and beta_channel not in dfw.channels:
            alpha = 1 + dfw._dataframe[channel] * dfw._dataframe[BetaDistribution.COUNT_CHANNEL]
            beta = 1 + (1 - dfw._dataframe[channel]) * dfw._dataframe[BetaDistribution.COUNT_CHANNEL]
            dfw._dataframe = dfw._dataframe.join(pd.DataFrame({alpha_channel: alpha, beta_channel: beta}))
        return new_channels


