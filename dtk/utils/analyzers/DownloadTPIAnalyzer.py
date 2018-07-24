import os
from simtools.Analysis.BaseAnalyzers import BaseCacheAnalyzer


class DownloadTPIAnalyzer(BaseCacheAnalyzer):
    """
    Similar to DownloadAnalyzer, but not quite, as the output directories need to be the exp_name and
    all sim results are sorted into subdirectories by their associated requested filename.
    """
    DONE = True

    def __init__(self, filenames, TPI_tag='TPI', working_dir="output"):
        super().__init__(cache_location=os.path.join(working_dir, "cache"))
        self.filenames = filenames
        self.parse = False
        self.TPI_tag = TPI_tag
        self.working_dir = working_dir

    def per_experiment(self, experiment):
        """
        Set and create the output path. Needs to be called before apply() on any of the sims AND
        after the experiments are known (dirname depends on experiment name)
        :param experiment: experiment object to make output directory for
        :return: Nothing
        """
        output_path = os.path.join(self.working_dir, experiment.exp_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for f in self.filenames:
            subdirectory, _ = os.path.splitext(os.path.basename(f))

            if not os.path.exists(os.path.join(output_path, subdirectory)):
                os.makedirs(os.path.join(output_path, subdirectory))

    def filter(self, simulation):
        """
        Determines if the given simulation should be downloaded
        :param simulation:
        :return: True/False : True if sim should be downloaded, False otherwise
        """
        return not self.is_in_cache(key=simulation.id)

    def select_simulation_data(self, data, simulation):
        output_dir = os.path.join(self.working_dir, simulation.experiment.exp_name)
        paths = []
        # Create the requested files
        for source_filename in self.filenames:
            # construct the full destination filename
            subdirectory, _ = os.path.splitext(os.path.basename(source_filename))
            dest_filename = self._construct_filename(simulation, source_filename)

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

    def _construct_filename(self, simulation, filename):
        # create the infix filename string e.g. TPI14_REP1, where the TPI number is the ordered sim number
        try:
            tpi_number = simulation.tags[self.TPI_tag]
            run_number = simulation.tags['Run_Number']
        except KeyError:
            raise KeyError('Experiment simulations must have the tag \'%s\' in order to be compatible with '
                           'DownloadAnalyzerTPI' % self.TPI_tag)
        infix_string = '_'.join(['TPI{:04d}'.format(int(tpi_number)), "REP{:04d}".format(run_number)])
        prefix, extension = os.path.splitext(filename)

        constructed_filename = '_'.join([prefix, infix_string]) + extension

        return constructed_filename