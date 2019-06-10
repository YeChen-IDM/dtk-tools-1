import logging
import os
import numpy as np
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import betainc
from examples.Separatrix.Algorithms.AlgoHelper.utils import find

logger = logging.getLogger(__name__)


def plotInference1D(inference_x, sample_x, new_sample_x, sample_y, new_sample_y, a, b, ParameterRanges,
                    true_separatrix, params, directory):
    Interest_Level = params["Interest_Level"]

    PMFBinEdges = np.linspace(0, 1, 100).T
    PMFBinCenters = PMFBinEdges[0:PMFBinEdges.shape[0] - 1] + (PMFBinEdges[1] - PMFBinEdges[0]) / 2

    pdf = np.zeros((99, params["Inference_Grid_Resolution"]))
    for j in range(0, params["Inference_Grid_Resolution"]):
        tmp = betainc(a[j], b[j], PMFBinEdges[1:, ])
        pdf[:, j] = np.append(tmp[0], np.diff(tmp, n=1, axis=0))

    [xx, yy] = np.meshgrid(inference_x, (PMFBinCenters - PMFBinCenters[0]) / (PMFBinCenters[-1] - PMFBinCenters[0]))

    # create a grid picture
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

        plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

        plt.ylim([-.10, 1.10])

        cbar = plt.colorbar(surf)

    plot_main(true_separatrix)

    # plt.show()

    # plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.pdf'))
    plt.savefig(os.path.join(directory, 'Separatrix_Sample_Results.png'))

    fig.clf()
    plt.close(fig)
