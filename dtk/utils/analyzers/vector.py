import os
from itertools import cycle, islice, repeat, chain

import pandas as pd

from dtk.utils.analyzers import default_filter_fn, default_select_fn, default_group_fn
from timeseries import TimeseriesAnalyzer


def default_vectorplot_fn(df, ax):
    grouped = df.groupby(level=['species', 'group'], axis=1)
    m = grouped.mean()
    nspecies, ngroups = map(len, m.keys().levels)
    colors = list(
        chain(*[list(repeat(c, ngroups)) for c in islice(cycle(['navy', 'firebrick', 'green']), None, nspecies)]))
    m.plot(ax=ax, legend=True, color=colors, alpha=0.5)


class VectorSpeciesAnalyzer(TimeseriesAnalyzer):
    plot_name = 'VectorChannelPlots'
    data_group_names = ['group', 'sim_id', 'species', 'channel']
    ordered_levels = ['channel', 'species', 'group', 'sim_id']
    output_file = 'vector.csv'

    def __init__(self,
                 filename=os.path.join('output', 'VectorSpeciesReport.json'),
                 filter_function=default_filter_fn,  # no filtering based on metadata
                 select_function=default_select_fn,  # return complete-&-unaltered timeseries
                 group_function=default_group_fn,  # group by unique simid-key from parser
                 plot_function=default_vectorplot_fn,
                 channels=['Adult Vectors Per Node', 'Percent Infectious Vectors', 'Daily EIR'],
                 saveOutput=False):
        TimeseriesAnalyzer.__init__(self, filename,
                                    filter_function, select_function,
                                    group_function, plot_function,
                                    channels, saveOutput)

    def get_channel_data(self, data_by_channel, header):
        species_channel_data = {}
        species_names = header["Subchannel_Metadata"]["MeaningPerAxis"][0][0]  # ?
        for i, species in enumerate(species_names):
            species_channel_series = [self.select_function(data_by_channel[channel]["Data"][i]) for channel in
                                      self.channels]
            species_channel_data[species] = pd.concat(species_channel_series, axis=1, keys=self.channels)
        return pd.concat(species_channel_data, axis=1)
