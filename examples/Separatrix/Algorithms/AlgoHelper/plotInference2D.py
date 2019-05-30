import json
import logging
import os
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf

from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection
from examples.Separatrix.Algorithms.AlgoHelper.SeparatrixInference import SeparatrixInference
from examples.Separatrix.Algorithms.AlgoHelper.tanhModel import tanhModel
from examples.Separatrix.Algorithms.ModelNextPoint import ModelNextPoint

logger = logging.getLogger(__name__)


def plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, a, b, ParameterRanges,
                    true_separatrix, params, directory):
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

        # qcs = ax1.contour(ix, iy, iz, levels=[iso], colors=['k'], linestyles='solid', norm=None)

        # qcs.collections[0].set_label('True')
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

    plt.show()

    # plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.pdf'))

    fig.clf()
    plt.close(fig)



def test1():
    # Load Separatrix settings
    Settings = json.load(open('../../Settings.json', 'r'))

    ParameterRanges = [dict(Min=0, Max=1), dict(Min=0, Max=1)]
    grid_res = Settings["Inference_Grid_Resolution"]

    np.random.seed(1)
    myrng = np.random.rand()
    model = tanhModel(myrng=myrng)

    sample_x = LHSPointSelection(Settings['Num_Initial_Samples'], NumDimensions=2, ParameterRanges=ParameterRanges)
    sample_y = model.Sample(sample_x)

    new_sample_x = LHSPointSelection(Settings['Num_Initial_Samples'], NumDimensions=2, ParameterRanges=ParameterRanges)
    new_sample_y = model.Sample(new_sample_x)

    # Make data.
    ix, iy = np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))
    inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T

    alpha, beta = SeparatrixInference(sample_x, sample_y, inference_x, Settings)

    iso = 0.7
    true_separatrix_qcs = model.TrueSeparatrix(iso)
    # qcs = true_separatrix_qcs

    # clear up the existing one!
    plt.clf()
    plt.cla()
    plt.close()

    plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta, ParameterRanges,
                    true_separatrix_qcs, Settings, directory='.')



def test2():
    # Load Separatrix settings
    Settings = json.load(open('../../Settings.json', 'r'))

    ParameterRanges = [dict(Min=0, Max=1), dict(Min=0, Max=1)]
    grid_res = Settings["Inference_Grid_Resolution"]

    np.random.seed(1)
    myrng = np.random.rand()
    model = tanhModel(myrng=myrng)

    test_case = False

    # Test #1
    if test_case:
        Settings["Num_Initial_Samples"] = 50      # if less than 30, Cause warning later!
        sample_x = LHSPointSelection(Settings['Num_Initial_Samples'], NumDimensions=2, ParameterRanges=ParameterRanges)
        sample_y = model.Sample(sample_x)

        # new_sample_x = LHSPointSelection(Settings['Num_Initial_Samples'], NumDimensions=2, ParameterRanges=ParameterRanges)
        # new_sample_y = model.Sample(new_sample_x)

        new_sample_x = None
        new_sample_y = None

    # Test #2
    if not test_case:
        params = [
            {
                'Name': 'Point_X',
                'MapTo': 'Point_X',
                'Min': 0,
                'Max': 1
            },
            {
                'Name': 'Point_Y',
                'MapTo': 'Point_Y',
                'Min': 0,
                'Max': 1
            },
        ]

        model_next_point = ModelNextPoint(params, Num_Dimensions=2, Num_Initial_Samples=40, Num_Next_Samples=20,
                                          Settings=Settings)

        # Settings["Num_Initial_Samples"] = 20
        # sample_x = model_next_point.get_lhs_samples(20)
        sample_x = LHSPointSelection(Settings['Num_Initial_Samples'], NumDimensions=2, ParameterRanges=ParameterRanges)
        sample_y = model.Sample(sample_x)

        # new_sample_x = model_next_point.get_lhs_samples(20)
        # new_sample_y = model.Sample(new_sample_x)
        new_sample_x = None
        new_sample_y = None




    # Make data.
    ix, iy = np.meshgrid(np.linspace(ParameterRanges[0]['Min'], ParameterRanges[0]['Max'], grid_res),
                         np.linspace(ParameterRanges[1]['Min'], ParameterRanges[1]['Max'], grid_res))
    inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T

    alpha, beta = SeparatrixInference(sample_x, sample_y, inference_x, Settings)

    iso = Settings["Interest_Level"]  # 0.7
    true_separatrix_qcs = model.TrueSeparatrix(iso)
    # qcs = true_separatrix_qcs

    # clear up the existing one!
    plt.clf()
    plt.cla()
    plt.close()

    plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, alpha, beta, ParameterRanges,
                    true_separatrix_qcs, Settings, directory='.')



if __name__ == "__main__":
    # test1()

    test2()
    exit()

