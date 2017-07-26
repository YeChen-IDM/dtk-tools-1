import json
import os

from dtk.utils.analyzers.BaseAnalyzer import BaseAnalyzer

class DownloadAnalyzerTPI(BaseAnalyzer):
    """
    Similar to DownloadAnalyzer, but not quite, as the output directories need to be the exp_name and
    all sim results are dropped into this flat directory.
    """
    TPI_tag = 'TPI'

    def __init__(self, filenames):
        super(DownloadAnalyzerTPI, self).__init__()
        self.output_path = None # we need to make sure this is set via per_experiment before calling self.apply
        self.filenames = filenames

    def initialize(self):
        pass

    def per_experiment(self, experiment):
        """
        Set and create the output path. Needs to be called before apply() on any of the sims AND
        after the experiments are known (dirname depends on experiment name)
        :param experiment: experiment object to make output directory for
        :return: Nothing
        """
        self.output_path = os.path.join(self.working_dir, experiment.exp_name)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def apply(self, parser):
        sim_folder = self.output_path # all sims for the exp in one directory
        # Create the requested files
        for source_filename in self.filenames:
            # construct the full destination filename
            dest_filename = self._construct_filename(parser, source_filename)

            file_path = os.path.join(sim_folder, os.path.basename(dest_filename))
            with open(file_path, 'wb') as outfile:
                if not isinstance(parser.raw_data[source_filename], str):
                    outfile.write(json.dumps(parser.raw_data[source_filename]))
                else:
                    outfile.write(parser.raw_data[source_filename])

    def _construct_filename(self, parser, filename):
        # create the infix filename string e.g. TPI14_REP1, where the TPI number is the ordered sim number
        try:
            tpi_number = parser.sim_data[self.TPI_tag]
        except KeyError:
            raise KeyError('Experiment simulations must have the tag \'%s\' in order to be compatible with '
                           'DownloadAnalyzerTPI' % self.TPI_tag)
        infix_string = '_'.join(['TPI%s' % tpi_number, 'REP1'])  # REPn is hardcoded for now; will need to change
        prefix, extension = os.path.splitext(filename)
        constructed_filename = '_'.join([prefix, infix_string]) + extension
        return constructed_filename