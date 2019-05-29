import logging
import os

from examples.Separatrix.Algorithms.AlgoHelper.tanhModel import tanhModel

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibtool.IterationState import IterationState
from calibtool.plotters.BasePlotter import BasePlotter
from calibtool.utils import StatusPoint

import numpy as np
from scipy.interpolate import Rbf

from scipy.special import betainc
from examples.Separatrix.Algorithms.AlgoHelper.SeparatrixInference import SeparatrixInference
from examples.Separatrix.Algorithms.AlgoHelper.utils import find

from examples.Separatrix.Algorithms.AlgoHelper.SigmoidalModel import SigmoidalModel

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

        data_this_iter = self.data.set_index('Iteration').loc[self.iteration_state.iteration]

        latest_results = data_this_iter['Results'].values  # Sort by sample?

        ### VIOLIN PLOTS BY ITERATION ###
        all_results = self.all_results.copy().reset_index(drop=True)
        fig, ax = plt.subplots()

        directory = self.iteration_state.iteration_directory

        self.plot_separatrix(directory)

    def plot_separatrix_bk(self, directory):
        # This import registers the 3D projection, but is otherwise unused.

        model_algo = self.iteration_state.next_point_algo
        Settings = model_algo.Settings

        ParameterRanges = model_algo.parameter_ranges  # [dict(Min=0, Max=1), dict(Min=0, Max=1)]

        grid_res = model_algo.Settings["Inference_Grid_Resolution"]

        if self.iteration_state.iteration == 0:
            sample_x, sample_y = model_algo.convert_df_to_points(self.iteration_state.iteration)
            new_sample_x = None
            new_sample_y = None
        else:
            sample_x, sample_y = model_algo.convert_df_to_points(self.iteration_state.iteration - 1)
            new_sample_x, new_sample_y = model_algo.convert_df_to_points(self.iteration_state.iteration)

        np.random.seed(1)
        myrng = np.random.rand()
        model = tanhModel(myrng=myrng)

        iso = 0.7
        true_separatrix_qct = model.TrueSeparatrix(iso)
        qcs = true_separatrix_qct

        # clear up the existing one!
        plt.clf()
        plt.cla()
        plt.close()

        # create a grid picture
        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        ax3 = plt.subplot2grid((2, 3), (1, 2))

        # Make data.
        ix, iy = np.meshgrid(np.linspace(ParameterRanges[0]['Min'], ParameterRanges[0]['Max'], grid_res),
                             np.linspace(ParameterRanges[1]['Min'], ParameterRanges[1]['Max'], grid_res))
        inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T
        # print("inference_x:\n", inference_x)

        X = inference_x[:, 0]
        Y = inference_x[:, 1]

        alpha, beta = SeparatrixInference(sample_x, sample_y, inference_x, model_algo.Settings)

        Z = (alpha - 1) / (alpha + beta - 2)
        Z2 = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

        Z = np.array([[0.00449804],
                      [0.00684196],
                      [0.01015733],
                      [0.01454983],
                      [0.01979778],
                      [0.02525712],
                      [0.03017686],
                      [0.03416047],
                      [0.03721998],
                      [0.03952568],
                      [0.00849735],
                      [0.01290589],
                      [0.01918379],
                      [0.02764459],
                      [0.03805234],
                      [0.04926124],
                      [0.0595879],
                      [0.0678479],
                      [0.07384079],
                      [0.07794589],
                      [0.01517875],
                      [0.02307668],
                      [0.03434587],
                      [0.04963031],
                      [0.06869854],
                      [0.08968128],
                      [0.10940491],
                      [0.12521209],
                      [0.13631278],
                      [0.14335516],
                      [0.02524223],
                      [0.03855026],
                      [0.05751103],
                      [0.08312536],
                      [0.11498775],
                      [0.15013506],
                      [0.18343786],
                      [0.21028774],
                      [0.22897424],
                      [0.24038946],
                      [0.03846403],
                      [0.05923495],
                      [0.08874151],
                      [0.12813027],
                      [0.17625387],
                      [0.22844435],
                      [0.27746199],
                      [0.31702264],
                      [0.3447146],
                      [0.36168467],
                      [0.05318521],
                      [0.08280918],
                      [0.12475561],
                      [0.17984187],
                      [0.24533131],
                      [0.31426227],
                      [0.37763423],
                      [0.42851722],
                      [0.46458624],
                      [0.48738527],
                      [0.06691679],
                      [0.10534424],
                      [0.15951796],
                      [0.22938595],
                      [0.31004327],
                      [0.39226254],
                      [0.4661328],
                      [0.52515357],
                      [0.56771891],
                      [0.59582797],
                      [0.07779716],
                      [0.12350837],
                      [0.1874661],
                      [0.26842444],
                      [0.35949073],
                      [0.45004566],
                      [0.53026041],
                      [0.59453853],
                      [0.64196377],
                      [0.67476256],
                      [0.08537441],
                      [0.13611478],
                      [0.20627627],
                      [0.29344546],
                      [0.38957021],
                      [0.48386789],
                      [0.56729442],
                      [0.63505246],
                      [0.68645862],
                      [0.72357188],
                      [0.09018875],
                      [0.1437542],
                      [0.21670177],
                      [0.30577668],
                      [0.40274316],
                      [0.49760217],
                      [0.58235435],
                      [0.65268131],
                      [0.70768366],
                      [0.74891464]])  # (a-1)./(a+b-2)
        Z2 = np.array([[0.0829739],
                       [0.06436369],
                       [0.04644937],
                       [0.06037986],
                       [0.05090435],
                       [0.05107968],
                       [0.06575824],
                       [0.07029055],
                       [0.06667271],
                       [0.08303053],
                       [0.05755405],
                       [0.03736139],
                       [0.05833867],
                       [0.04209699],
                       [0.03022084],
                       [0.02610546],
                       [0.05666454],
                       [0.06948502],
                       [0.05087092],
                       [0.08036032],
                       [0.05481178],
                       [0.01993541],
                       [0.04471436],
                       [0.0485776],
                       [0.03724029],
                       [0.04652495],
                       [0.07011047],
                       [0.06927966],
                       [0.05187523],
                       [0.06804544],
                       [0.05479329],
                       [0.05364593],
                       [0.02935156],
                       [0.0587021],
                       [0.05235969],
                       [0.0604109],
                       [0.04564273],
                       [0.04016312],
                       [0.0392023],
                       [0.07666595],
                       [0.05740067],
                       [0.05960978],
                       [0.04359631],
                       [0.04696039],
                       [0.06809307],
                       [0.06958616],
                       [0.04333693],
                       [0.03706743],
                       [0.04242826],
                       [0.06444016],
                       [0.08102545],
                       [0.08056711],
                       [0.0521463],
                       [0.02704309],
                       [0.03950543],
                       [0.07759123],
                       [0.06224356],
                       [0.04485535],
                       [0.05233009],
                       [0.05583938],
                       [0.06089449],
                       [0.04215777],
                       [0.04908664],
                       [0.04728356],
                       [0.04621789],
                       [0.05521787],
                       [0.05797472],
                       [0.05537478],
                       [0.05350288],
                       [0.06861179],
                       [0.05653671],
                       [0.03501611],
                       [0.04766386],
                       [0.05625185],
                       [0.07307217],
                       [0.06094678],
                       [0.04883582],
                       [0.04394175],
                       [0.06467802],
                       [0.067456],
                       [0.06402629],
                       [0.05986484],
                       [0.0548994],
                       [0.06333494],
                       [0.05144072],
                       [0.04378508],
                       [0.04679338],
                       [0.06102855],
                       [0.06571524],
                       [0.05246935],
                       [0.07056794],
                       [0.08096526],
                       [0.07564897],
                       [0.08110097],
                       [0.07886796],
                       [0.05768439],
                       [0.05114243],
                       [0.07091129],
                       [0.05344126],
                       [0.06257405]])  # a.*b./((a+b).^2 .* (a+b+1))

        Z = (alpha - 1) / (alpha + beta - 2)
        Z2 = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

        # not used and use pcolormesh instead
        # triang = mtri.Triangulation(X, Y)

        rbf = Rbf(X, Y, Z, function='linear')
        iz = rbf(ix, iy)

        rbf2 = Rbf(X, Y, Z2, function='linear')
        iz2 = rbf2(ix, iy)

        def plot_main(qcs):

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

            # ax3.legend([h1, h2], labels=['Success', 'Fail'], loc='lower center')
            # ax3.legend([h1, h2], labels=['Success', 'Fail'], loc='lower left', bbox_to_anchor=(0., -0.14, 1., .102), ncol=3, mode="expand", borderaxespad=0.)  # [TODO]: Cause warning!
            ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

        plot_main(qcs)
        plot_variance(qcs)
        plot_samples(sample_x, sample_y, new_sample_x, new_sample_y)

        # plt.show()

        plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.pdf'))

        fig.clf()
        plt.close(fig)

    def plot_separatrix(self, directory):
        # This import registers the 3D projection, but is otherwise unused.

        model_algo = self.iteration_state.next_point_algo
        Settings = model_algo.Settings
        Num_Dimensions = model_algo.Num_Dimensions

        ParameterRanges = model_algo.parameter_ranges  # [dict(Min=0, Max=1), dict(Min=0, Max=1)]

        grid_res = model_algo.Settings["Inference_Grid_Resolution"]

        if self.iteration_state.iteration == 0:
            sample_x, sample_y = model_algo.convert_df_to_points(self.iteration_state.iteration)
            new_sample_x = None
            new_sample_y = None
        else:
            sample_x, sample_y = model_algo.convert_df_to_points(self.iteration_state.iteration - 1)
            new_sample_x, new_sample_y = model_algo.convert_df_to_points(self.iteration_state.iteration)

        np.random.seed(1)
        myrng = np.random.rand()

        if Num_Dimensions == 1:
            model = SigmoidalModel(myrng=myrng)
        elif Num_Dimensions == 2:
            model = tanhModel(myrng=myrng)

        iso = Settings["Interest_Level"]  # 0.7
        true_separatrix_qcs = model.TrueSeparatrix(iso)
        # qcs = true_separatrix_qcs

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

        # print("inference_x:\n", inference_x)

        # clear up the existing one!
        plt.clf()
        plt.cla()
        plt.close()

        if Num_Dimensions == 1:
            self.plotInference1D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta,
                                 ParameterRanges, true_separatrix_qcs, Settings)
        elif Num_Dimensions == 2:
            self.plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta,
                                 ParameterRanges, true_separatrix_qcs, Settings)

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
        # print(xx)
        # print(yy)

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
            print('plot_variance, iso: ', iso)

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

            # ax3.legend([h1, h2], labels=['Success', 'Fail'], loc='lower center')
            # ax3.legend([h1, h2], labels=['Success', 'Fail'], loc='lower left', bbox_to_anchor=(0., -0.14, 1., .102), ncol=3, mode="expand", borderaxespad=0.)  # [TODO]: Cause warning!
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
