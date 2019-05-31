import logging
import os
import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.special import betainc
from calibtool.utils import StatusPoint
from calibtool.IterationState import IterationState
from calibtool.plotters.BasePlotter import BasePlotter
from examples.Separatrix.Algorithms.AlgoHelper.utils import find
from examples.Separatrix.Algorithms.AlgoHelper.SigmoidalModel import SigmoidalModel
from examples.Separatrix.Algorithms.AlgoHelper.tanhModel import tanhModel
from examples.Separatrix.Algorithms.AlgoHelper.SeparatrixInference import SeparatrixInference

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

        # self.directory = self.iteration_state.iteration_directory
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

        directory = self.iteration_state.iteration_directory
        self.plot_separatrix(directory)

    def plot_separatrix(self, directory):
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

        if Num_Dimensions == 1:
            self.plotInference1D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta,
                                 ParameterRanges, true_separatrix_qcs, Settings)
        elif Num_Dimensions == 2:
            self.plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta,
                                 ParameterRanges, true_separatrix_qcs, Settings)

    # This is the same as in file: examples/Separatrix/Algorithms/AlgoHelper/plotInference1D.py
    def plotInference1D(self, inference_x, sample_x, new_sample_x, sample_y, new_sample_y, a, b, ParameterRanges,
                        true_separatrix, params):

        Interest_Level = params["Interest_Level"]

        PMFBinEdges = np.linspace(0, 1, 100).T
        PMFBinCenters = PMFBinEdges[0:PMFBinEdges.shape[0] - 1] + (PMFBinEdges[1] - PMFBinEdges[0]) / 2

        pdf = np.zeros((99, params["Inference_Grid_Resolution"]))
        for j in range(0, params["Inference_Grid_Resolution"]):
            tmp = betainc(a[j], b[j], PMFBinEdges[1:, ])
            pdf[:, j] = np.append(tmp[0], np.diff(tmp, n=1, axis=0))

        [xx, yy] = np.meshgrid(inference_x, (PMFBinCenters - PMFBinCenters[0]) / (PMFBinCenters[-1] - PMFBinCenters[0]))

        # create a grid picture
        # fig = plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 8))

        def plot_main(true_separatrix):

            surf = ax.pcolormesh(xx, yy, pdf, cmap='viridis', shading='gouraud')  # smooth

            h_mean = ax.plot(inference_x, a / (a + b), 'k:', linewidth=4, label='Mean')
            h_mode = ax.plot(inference_x, (a - 1) / (a + b - 2), 'k-', linewidth=4, label='Mode')

            h_separatrix = ax.plot(true_separatrix * np.ones((1, 2)).flatten(), np.array([0, 1]), 'm-', linewidth=4,
                                   label='True Separatrix')

            h_interestLevel = ax.plot(np.array([ParameterRanges[0]['Min'], ParameterRanges[0]['Max']]),
                                      Interest_Level * np.ones((1, 2)).flatten(), 'c-', linewidth=4,
                                      label='Interest Level')

            inds = find(sample_y)
            h_success = ax.plot(sample_x[inds, 0].reshape(len(inds[0]), 1), 1.05 * np.ones((len(inds[0]), 1)), 'k+',
                                markersize=12, label='Success')

            inds = find(sample_y == 0)
            h_fail = ax.plot(sample_x[inds, 0].reshape(len(inds[0]), 1), -0.05 * np.ones((len(inds[0]), 1)), 'ko',
                             markersize=12, label='Fail')

            if new_sample_y is not None and len(new_sample_y) > 0:
                inds = find(new_sample_y)
                h3 = ax.plot(new_sample_x[inds, 0].reshape(len(inds[0]), 1), 1.05 * np.ones((len(inds[0]), 1)), 'r+',
                             markersize=12, label='New Success')

                inds = find(new_sample_y == 0)
                h4 = ax.plot(new_sample_x[inds, 0].reshape(len(inds[0]), 1), -0.05 * np.ones((len(inds[0]), 1)), 'ro',
                             markersize=12, label='New Fail')

            ax.set_xlabel('X')
            ax.set_ylabel('Probability Density')
            # ax.set_title('Samples')

            # plt.legend(loc='upper left')
            plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

            plt.ylim([-.10, 1.10])

            cbar = plt.colorbar(surf)

        plot_main(true_separatrix)

        # plt.show()

        directory = self.iteration_state.iteration_directory
        plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.pdf'))

        fig.clf()
        plt.close(fig)

    # This is the same as in file: examples/Separatrix/Algorithms/AlgoHelper/plotInference2D.py
    def plotInference2D(self, inference_x, sample_x, new_sample_x, sample_y, new_sample_y, a, b, ParameterRanges,
                        true_separatrix, params):
        # create a grid picture
        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax3 = plt.subplot2grid((2, 3), (1, 2))

        iso = params["Interest_Level"]

        # Make data.
        grid_res = params["Inference_Grid_Resolution"]
        ix, iy = np.meshgrid(np.linspace(ParameterRanges[0]['Min'], ParameterRanges[0]['Max'], grid_res),
                             np.linspace(ParameterRanges[1]['Min'], ParameterRanges[1]['Max'], grid_res))
        inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T
        # print("inference_x:\n", inference_x)

        X = inference_x[:, 0]
        Y = inference_x[:, 1]

        Z = (a - 1) / (a + b - 2)
        Z2 = a * b / ((a + b) ** 2 * (a + b + 1))

        rbf = Rbf(X, Y, Z, function='linear')
        iz = rbf(ix, iy)

        rbf2 = Rbf(X, Y, Z2, function='linear')
        iz2 = rbf2(ix, iy)

        def plot_main(qcs):

            iso = params["Interest_Level"]
            surf = ax1.pcolormesh(ix, iy, iz, cmap='viridis', shading='gouraud')  # smooth

            cp = None
            for j in range(len(qcs.allsegs)):
                for ii, seg in enumerate(qcs.allsegs[j]):
                    if cp is None:
                        cp = seg
                    else:
                        cp = np.vstack((cp, seg))

            h = ax1.plot(cp[:, 0], cp[:, 1], '--', color='black', label='Estimate')

            qcs = ax1.contour(ix, iy, iz, levels=[iso], colors=['k'], linestyles='solid', norm=None)

            qcs.collections[0].set_label('True')
            ax1.legend(loc='lower left')
            # plt.legend([h, qcs], labels=['Estimate', 'True'])

            ax1.set_title('Mode of Success Probability')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')

            cbar = plt.colorbar(surf, ax=ax1)

        def plot_variance(qcs):

            iso = params["Interest_Level"]
            surf = ax2.pcolormesh(ix, iy, iz2, cmap='viridis', shading='gouraud')  # smooth

            cp = None
            for j in range(len(qcs.allsegs)):
                for ii, seg in enumerate(qcs.allsegs[j]):
                    if cp is None:
                        cp = seg
                    else:
                        cp = np.vstack((cp, seg))

            h = ax2.plot(cp[:, 0], cp[:, 1], '--', color='black', label='Estimate')

            qcs = ax2.contour(ix, iy, iz, levels=[iso], colors=['k'], linestyles='solid', norm=None)

            # qcs.collections[0].set_label('True')
            # ax2.legend(loc='lower left')

            ax2.set_title('Variacne')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

            # cbar = plt.colorbar(surf, ax=ax2)

        def plot_samples(sample_x, sample_y, new_sample_x=None, new_sample_y=None):

            sample_y = sample_y.ravel()

            success = sample_y > 0
            fail = sample_y < 1

            h1 = ax3.scatter(sample_x[success, 0], sample_x[success, 1], c='k', marker='+', label='Success')
            h2 = ax3.scatter(sample_x[fail, 0], sample_x[fail, 1], c='k', marker='o', label='Fail')

            if new_sample_y is not None and len(new_sample_y) > 0:
                new_sample_y = new_sample_y.ravel()
                new_success = new_sample_y > 0
                new_fail = new_sample_y < 1

                h3 = ax3.scatter(new_sample_x[new_success, 0], new_sample_x[new_success, 1], c='r', marker='+',
                                 label='New Success')
                h4 = ax3.scatter(new_sample_x[new_fail, 0], new_sample_x[new_fail, 1], c='r', marker='o',
                                 label='New Fail')

            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_title('Samples')

            ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

        qcs = true_separatrix
        plot_main(qcs)
        plot_variance(qcs)
        plot_samples(sample_x, sample_y, new_sample_x, new_sample_y)

        # plt.show()

        directory = self.iteration_state.iteration_directory
        plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.pdf'))

        fig.clf()
        plt.close(fig)

    def cleanup(self):
        """
        cleanup the existing plots
        :param calib_manager:
        :return:
        """
        pass
