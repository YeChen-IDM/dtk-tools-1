import logging
import os
import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

logger = logging.getLogger(__name__)


def plotInference2D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, a, b, ParameterRanges,
                    true_separatrix, params, directory):

    # create a grid picture
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))

    # Make data.
    grid_res = 100
    xx, yy = np.meshgrid(np.linspace(ParameterRanges[0]['Min'], ParameterRanges[0]['Max'], grid_res),
                         np.linspace(ParameterRanges[1]['Min'], ParameterRanges[1]['Max'], grid_res))

    X = inference_x[:, 0]
    Y = inference_x[:, 1]

    Z = (a - 1) / (a + b - 2)
    Z2 = a * b / ((a + b) ** 2 * (a + b + 1))

    # [TODO]: what is the decision?
    Z[np.isneginf(Z)] = 0
    Z[np.isinf(Z)] = 0
    Z[np.isnan(Z)] = 0

    # [TODO]: what is the decision?
    Z2[np.isneginf(Z2)] = 0
    Z2[np.isinf(Z2)] = 0
    Z2[np.isnan(Z2)] = 0

    # Approach #1: use Rbf
    rbf = Rbf(X, Y, Z, function='linear')
    zz = rbf(xx, yy)

    rbf2 = Rbf(X, Y, Z2, function='linear')
    zz2 = rbf2(xx, yy)

    # Approach #2: use interp2d
    # x = np.linspace(ParameterRanges[0]['Min'], ParameterRanges[0]['Max'], grid_res)
    # y = np.linspace(ParameterRanges[1]['Min'], ParameterRanges[1]['Max'], grid_res)
    #
    # f = interp2d(X, Y, Z, kind='quintic')  # ‘linear’, ‘cubic’, ‘quintic’}, optional
    # zz = f(x, y)
    #
    # f = interp2d(X, Y, Z2, kind='quintic')  # ‘linear’, ‘cubic’, ‘quintic’}, optional
    # zz2 = f(x, y)

    def plot_main(qcs):

        iso = params["Interest_Level"]
        surf = ax1.pcolormesh(xx, yy, zz, cmap='viridis', shading='gouraud')  # smooth

        cp = None
        for j in range(len(qcs.allsegs)):
            for ii, seg in enumerate(qcs.allsegs[j]):
                if cp is None:
                    cp = seg
                else:
                    cp = np.vstack((cp, seg))

        h = ax1.plot(cp[:, 0], cp[:, 1], '--', color='black', label='Estimate')

        qcs = ax1.contour(xx, yy, zz, levels=[iso], colors=['k'], linestyles='solid', norm=None)

        ax1.legend(loc='lower left')

        ax1.set_title('Mode of Success Probability')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        cbar = plt.colorbar(surf, ax=ax1)

    def plot_variance(qcs):

        iso = params["Interest_Level"]
        surf = ax2.pcolormesh(xx, yy, zz2, cmap='viridis', shading='gouraud')  # smooth

        cp = None
        for j in range(len(qcs.allsegs)):
            for ii, seg in enumerate(qcs.allsegs[j]):
                if cp is None:
                    cp = seg
                else:
                    cp = np.vstack((cp, seg))

        h = ax2.plot(cp[:, 0], cp[:, 1], '--', color='black', label='Estimate')

        qcs = ax2.contour(xx, yy, zz, levels=[iso], colors=['k'], linestyles='solid', norm=None)

        ax2.set_title('Variance')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')


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

    # plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.pdf'))
    plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.png'))

    fig.clf()
    plt.close(fig)


