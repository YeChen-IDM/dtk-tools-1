import pandas as pd

from dtk.utils.observations.AgeBin import AgeBin
from dtk.utils.observations.DataFrameWrapper import DataFrameWrapper


class PopulationObs(DataFrameWrapper):

    PROVINCIAL = 'Provincial'
    NON_PROVINCIAL = 'Non-provincial'
    AGGREGATED_NODE = 0  # a reserved node number for non-provincial analysis
    AGGREGATED_PROVINCE = 'All'
    WEIGHT_CHANNEL = 'weight'

    def __init__(self, filename=None, dataframe=None, stratifiers=None):
        super().__init__(filename=filename, dataframe=dataframe, stratifiers=stratifiers)

        # calculations using the data should update this list after joining on self._dataframe
        self.derived_items = []
        self.adjusted_years = False

    def get_years(self):
        additional_required_stratifier = ['Year']
        self.verify_required_items(needed=additional_required_stratifier)
        return sorted(list(set(self._dataframe['Year'])))

    #
    # derived data computations
    #

    def fix_age_bins(self):
        """
        A method that converts the ', ' separated AgeBin format: [X, Y) to the new ':' format: [X:Y) for
        back-compatibility.
        :return: nothing
        """
        required_data = ['AgeBin']
        self.verify_required_items(needed=required_data)
        # self._dataframe['AgeBin'] = [age_bin.replace(', ', AgeBin.DEFAULT_DELIMITER)
        #                              for age_bin in self._dataframe['AgeBin']]
        new_bins = [age_bin.replace(', ', AgeBin.DEFAULT_DELIMITER) for age_bin in self._dataframe['AgeBin']]
        self._dataframe.assign(**{'AgeBin': new_bins})

    def get_age_bins(self):
        required_data = ['AgeBin']
        self.verify_required_items(needed=required_data)
        return list(self._dataframe['AgeBin'].unique())

    def get_provinces(self):
        required_data = ['Province']
        self.verify_required_items(needed=required_data)
        return list(self._dataframe['Province'].unique())

    def get_genders(self):
        required_data = ['Gender']
        self.verify_required_items(needed=required_data)
        return list(self._dataframe['Gender'].unique())

    def get_years(self):
        required_data = ['Year']
        self.verify_required_items(needed=required_data)
        return list(self._dataframe['Year'].unique())

    def adjust_years(self):
        if not self.adjusted_years:
            required_data = ['Year']
            self.verify_required_items(needed=required_data)
            self._dataframe = self._dataframe.assign(**{'Year': self._dataframe['Year']+0.5})
            self.adjusted_years = True

    @classmethod
    def construct_beta_channel(cls, channel, provinciality, age_bins, type):
        age_bins = age_bins if isinstance(age_bins, list) else [age_bins]
        age_bin_str = '_'.join([str(age_bin) for age_bin in age_bins])
        return '%s-%s-%s--Beta-%s' % (channel, provinciality, age_bin_str, type)

    def add_beta_parameters(self, channel, provinciality, age_bins):
        from dtk.utils.observations.BetaDistribution import BetaDistribution
        """
        Compute and add alpha, beta parameters for a beta distribution to the current self._dataframe object.
            Distribution is computed for the provided channel (data field), using 'count'. Result is put into new
            channels/columns named <channel>--Beta-alpha, <channel>--Beta-beta. If both alpha/beta channels already
            exist in the dataframe, nothing is computed.
        :param channel: The data channel/column to compute the beta distribution for.
        :return: a list of the channel-associated alpha and beta parameter channel names.
        """
        required_data = [BetaDistribution.COUNT_CHANNEL, channel]
        self.verify_required_items(needed=required_data)

        alpha_channel = self.construct_beta_channel(channel=channel, provinciality=provinciality, age_bins=age_bins,
                                                    type='alpha')
        beta_channel = self.construct_beta_channel(channel=channel, provinciality=provinciality, age_bins=age_bins,
                                                   type='beta')
        new_channels = [alpha_channel, beta_channel]

        # Useful for an 'omg what is going on!' type of check
        for ch in new_channels:
            if ch in self._dataframe.columns:
                raise Exception('Channel %s already exists in dataframe.' % ch)

        if alpha_channel not in self.channels and beta_channel not in self.channels:
            alpha = 1 + self._dataframe[channel] * self._dataframe[BetaDistribution.COUNT_CHANNEL]
            beta = 1 + (1 - self._dataframe[channel]) * self._dataframe[BetaDistribution.COUNT_CHANNEL]
            self._dataframe = self._dataframe.join(pd.DataFrame({alpha_channel: alpha, beta_channel: beta}))
            self.derived_items += new_channels
        return new_channels

    def add_beta_percentile_values(self, channel, provinciality, age_bins, p):
        """
        Computes the inverse beta distribution of 'value' at the specified probability threshold.
        :param channel: the channel/column with beta distribution parameters to compute percentiles with.
        :param p: probability threshold, float, 0-1
        :return: a list of the newly added (single) channel. Adds the column e.g. <channel>--Beta-0.025 (for 2.5000% threshold) for the designated threshold.
        """
        from scipy.stats import beta

        alpha_channel = self.construct_beta_channel(channel=channel, provinciality=provinciality, type='alpha')
        beta_channel = self.construct_beta_channel(channel=channel, provinciality=provinciality, type='beta')
        required_items = [alpha_channel, beta_channel]
        try:
            self.verify_required_items(needed=required_items)
        except self.MissingRequiredData:
            self.add_beta_parameters(channel=channel, provinciality=provinciality, age_bins=age_bins)

        values = beta.ppf(p, self._dataframe[alpha_channel], self._dataframe[beta_channel])
        p_channel = self.construct_beta_channel(channel=channel, provinciality=provinciality, type=p)
        values_df = pd.DataFrame({p_channel: values})
        self._dataframe = self._dataframe.join(values_df)
        new_channels = [p_channel]
        self.derived_items += new_channels
        return new_channels
