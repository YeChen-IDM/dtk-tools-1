import os

from simtools.Analysis.BaseAnalyzers import BaseCacheAnalyzer


class DownloadAnalyzerTPI(BaseCacheAnalyzer):
    """
    This analyzer is based on the DownloadAnalyzer and allows the download of files based on a Index and Repetition tags
    """

    def __init__(self, filenames, TPI_tag=None, REP_tag=None, output_path="output", force=False, ignore_REP=False, ignore_TPI=False):
        super().__init__(filenames=filenames, parse=False, force=force)
        self.TPI_tag = TPI_tag or "__sample_index__"
        self.REP_tag = REP_tag or "Run_Number"
        self.ignore_REP = ignore_REP
        self.ignore_TPI = ignore_TPI
        self.output_path = output_path

    def initialize(self):
        super().initialize()
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def per_experiment(self, experiment):
        """
        Set and create the output path.
        :param experiment: experiment object to make output directory for
        :return: Nothing
        """
        output_path = os.path.join(self.working_dir, self.output_path, experiment.exp_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for f in self.filenames:
            subdirectory, _ = os.path.splitext(os.path.basename(f))

            if not os.path.exists(os.path.join(output_path, subdirectory)):
                os.makedirs(os.path.join(output_path, subdirectory))

    def select_simulation_data(self, data, simulation):
        output_dir = os.path.join(self.working_dir, self.output_path, simulation.experiment.exp_name)
        if not os.path.exists(output_dir):
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
        paths = []
        # Create the requested files
        for source_filename in self.filenames:
            # construct the full destination filename
            dest_filename = self._construct_filename(simulation.tags, source_filename)
            subdirectory, _ = os.path.splitext(os.path.basename(source_filename))

            # sorting all downloaded results by requested filename
            file_path = os.path.join(output_dir, subdirectory, os.path.basename(dest_filename))
            paths.append(file_path)

            with open(file_path, 'wb') as outfile:
                try:
                    outfile.write(data[source_filename])
                except Exception as e:
                    print("Could not write the file {} for simulation {}".format(source_filename, simulation.id))

        # # now update the shelf/cache
        if all([os.path.exists(fp) for fp in paths]):
            self.to_cache(key=simulation.id, data=True)

    def _construct_filename(self, simulation_tags, filename):
        try:
            tpi_number = simulation_tags[self.TPI_tag] if not self.ignore_TPI else None
            run_number = simulation_tags[self.REP_tag] if not self.ignore_REP else None
        except KeyError:
            raise KeyError('Experiment simulations must have the following tags in order to be compatible: '
                           '\n- {} for the TPI index'
                           '\n- {} for the repetition'
                           '\n You can adjust those tags in the constructor of the analyzer.'
                           .format(self.TPI_tag, self.REP_tag))

        infix = []
        if not self.ignore_TPI:
            infix.append('TPI{:04d}'.format(int(tpi_number)))
        if not self.ignore_REP:
            infix.append("REP{:04d}".format(run_number))

        prefix, extension = os.path.splitext(filename)

        constructed_filename = '_'.join([prefix, *infix]) + extension

        return constructed_filename
