import logging
import os
import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibtool.utils import StatusPoint
from calibtool.IterationState import IterationState
from calibtool.plotters.BasePlotter import BasePlotter
from examples.Separatrix.Algorithms.AlgoHelper.tanhModel import tanhModel
from examples.Separatrix.Algorithms.AlgoHelper.SigmoidalModel import SigmoidalModel
from examples.Separatrix.Algorithms.AlgoHelper.SeparatrixInference import SeparatrixInference
from examples.Separatrix.Algorithms.AlgoHelper.plotInference1D import plotInference1D
from examples.Separatrix.Algorithms.AlgoHelper.plotInference2D import plotInference2D

logger = logging.getLogger(__name__)


class ModelPlotter(BasePlotter):
    def __init__(self, combine_sites=True):
        super(ModelPlotter, self).__init__(combine_sites)

    @property
    def directory(self):
        return self.get_plot_directory()

    # ZD [TODO]: self.iteration_state.analyzer_list doesn't keep site info, here we assume all analyzers have different names!!!
    def get_site_analyzer(self, site_name, analyzer_name):
        for site, analyzers in self.site_analyzer_names.items():
            if site_name != site:
                continue
            for analyzer in self.iteration_state.analyzer_list:
                if analyzer_name == analyzer.name:
                    return analyzer
        raise Exception('Unable to find analyzer=%s for site=%s' % (analyzer_name, site_name))

    def get_analyzer_data(self, iteration, site_name, analyzer_name):
        site_analyzer = '%s_%s' % (site_name, analyzer_name)
        return IterationState.restore_state(self.iteration_state.calibration_name, iteration).analyzers[site_analyzer]

    def visualize(self, iteration_state):
        self.iteration_state = iteration_state
        self.site_analyzer_names = iteration_state.site_analyzer_names
        iteration_status = self.iteration_state.status

        self.param_names = self.iteration_state.param_names
        self.data = self.iteration_state.next_point_algo.get_state()

        if iteration_status == StatusPoint.commission:
            if self.iteration_state.iteration > 0:
                pass
        elif iteration_status == StatusPoint.plot:
            self.visualize_results()
        else:
            raise Exception('Unknown stage %s' % iteration_status.name)

    def visualize_results(self):
        self.plot_separatrix()

    def plot_separatrix(self):
        model_algo = self.iteration_state.next_point_algo
        Settings = model_algo.Settings
        Num_Dimensions = model_algo.Num_Dimensions

        ParameterRanges = model_algo.parameter_ranges  # [dict(Min=0, Max=1), dict(Min=0, Max=1)]

        grid_res = model_algo.Settings["Inference_Grid_Resolution"]

        if self.iteration_state.iteration == 0:
            sample_x, sample_y = model_algo.get_sample_points(self.iteration_state.iteration)
            new_sample_x = None
            new_sample_y = None
        else:
            sample_x, sample_y = model_algo.get_sample_points(self.iteration_state.iteration - 1)
            new_sample_x, new_sample_y = model_algo.get_sample_points(self.iteration_state.iteration)

        np.random.seed(1)
        myrng = np.random.rand()

        if Num_Dimensions == 1:
            model = SigmoidalModel(myrng=myrng)
        elif Num_Dimensions == 2:
            model = tanhModel(myrng=myrng)

        iso = Settings["Interest_Level"]  # 0.7
        true_separatrix_qcs = model.TrueSeparatrix(iso)

        # Make data.
        if Num_Dimensions == 1:
            inference_x = np.meshgrid(np.linspace(0, 1, grid_res))
            inference_x = np.array(inference_x)
            inference_x = inference_x.reshape(inference_x.shape[1], 1)

            alpha, beta = SeparatrixInference(sample_x, sample_y, inference_x, Settings)
        elif Num_Dimensions == 2:
            ix, iy = np.meshgrid(np.linspace(ParameterRanges[0]['Min'], ParameterRanges[0]['Max'], grid_res),
                                 np.linspace(ParameterRanges[1]['Min'], ParameterRanges[1]['Max'], grid_res))
            inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T

            alpha, beta = SeparatrixInference(sample_x, sample_y, inference_x, Settings)

        # clear up the existing one!
        plt.cla()
        plt.clf()
        plt.close()

        directory = self.iteration_state.iteration_directory

        if Num_Dimensions == 1:
            plotInference1D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta,
                            ParameterRanges, true_separatrix_qcs, Settings, directory)
        elif Num_Dimensions == 2:
            plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta,
                            ParameterRanges, true_separatrix_qcs, Settings, directory)

    def cleanup(self):
        """
        cleanup the existing plots
        :param calib_manager:
        :return:
        """
        pass
