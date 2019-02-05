from typing import List, Optional, Mapping

from dtk.utils.observations.AgeBin import AgeBin
from dtk.utils.observations.DataFrameWrapper import DataFrameWrapper


class PopulationObs(DataFrameWrapper):
    PROVINCIAL = 'Provincial'
    NON_PROVINCIAL = 'Non-provincial'
    AGGREGATED_NODE = 0  # a reserved node number for non-provincial analysis
    AGGREGATED_PROVINCE = 'All'
    WEIGHT_CHANNEL = 'weight'

    def __init__(self, filename=None, dataframe=None, stratifiers=None, observations=None):
        super().__init__(filename=filename, dataframe=dataframe, stratifiers=stratifiers)

        # calculations using the data should update this list after joining on self._dataframe
        self.derived_items = []
        self.adjusted_years = False
        self.observations = observations

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
        return sorted(self._dataframe['Year'].unique())

    def adjust_years(self):
        if not self.adjusted_years:
            required_data = ['Year']
            self.verify_required_items(needed=required_data)
            self._dataframe = self._dataframe.assign(**{'Year': self._dataframe['Year'] + 0.5})
            self.adjusted_years = True

    def add_percentile_values(self, channel, distribution, p):
        """
        Computes the inverse beta distribution of 'value' at the specified probability threshold.
        :param channel: the channel/column with beta distribution parameters to compute percentiles with.
        :param p: probability threshold, float, 0-1
        :return: a list of the newly added (single) channel. Adds the column e.g. <channel>--Beta-0.025 (for 2.5000% threshold) for the designated threshold.
        """
        new_channels = distribution.add_percentile_values(dfw=self, channel=channel, p=p)
        self.derived_items += new_channels
        return new_channels

    def find_missing_tuples(self, target:object, columns_to_check: List[str] = ['AgeBin', 'Year', 'Gender']) -> Mapping[str, Optional[tuple]]:
        """
        Finds the missing tuples in the target.
        While the `is_included_in` function returns True or False if included or not, this slower function
        searches for all the missing tuples.

        Args:
            target: The target PopulationObs in which to check
            columns_to_check: Which columns are we basing the check

        Returns: Dict with key: observation (incidece, population) value: list of missing tuples for this observation
        None if nothing is missing
        """
        missing = {}
        base_db = self._dataframe
        target_df = target._dataframe

        for obs in self.observations:
            colums_to_keep = [*columns_to_check, obs]
            # Only consider observations where is not None and discard the rest
            left = base_db[base_db[obs].notnull()][colums_to_keep]
            right = target_df[target_df[obs].notnull()][colums_to_keep]

            # Merge the 2 dataframes
            merged_df = left.merge(right, how='left', on=columns_to_check, indicator=True)

            # Only keep the keys that are in the left one (our current object)
            left_only = merged_df[merged_df['_merge'] == "left_only"]
            if left_only.empty:
                continue

            # We had missing ones
            missing[obs] = [tuple(x) for x in left_only[columns_to_check].values]

        return missing

