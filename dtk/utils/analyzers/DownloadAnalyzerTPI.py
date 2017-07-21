import json
import os

from dtk.utils.analyzers.DownloadAnalyzer import DownloadAnalyzer

class DownloadAnalyzerTPI(DownloadAnalyzer):
    def __init__(self, experiment, filenames):
        super(DownloadAnalyzerTPI, self).__init__(filenames=filenames, output_path=experiment.exp_name)

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
            tpi_number = parser.sim_data['TPI'] # ck4, should be TPI in deployed code, CLEANUP AFTER DEBUG
        except KeyError:
            raise KeyError('Experiment simulations must have the tag \'TPI\' in order to be compatible with'
                           'DownloadAnalyzerTPI')
        infix_string = '_'.join(['TPI%s' % tpi_number, 'REP1'])  # REPn is hardcoded for now; will need to change
        prefix, extension = os.path.splitext(filename)
        constructed_filename = '_'.join([prefix, infix_string]) + extension
        return constructed_filename
