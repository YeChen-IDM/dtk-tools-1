import os
import numpy as np
import pandas as pd
from calibtool.resamplers.BaseResampler import BaseResampler
from calibtool.resamplers.CalibrationPoint import CalibrationPoint
from calibtool.algorithms.FisherInfMatrix import FisherInfMatrix, plot_cov_ellipse, trunc_gauss

import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA

class CramerRaoResampler(BaseResampler):
    def __init__(self, **kwargs):
        """
        :param n_resampling_points: The number of resampled points to generate
        :param kwargs: These are arguments passed directly to the underlying resampling routine.
        """
        super().__init__()
        # self.n_resampling_points = n_resampling_points # the number of points to resample/generate
        self.resample_kwargs = kwargs


    def resample(self, calibrated_points, selection_values, initial_calibration_points):
        """
        Takes in a list of 1+ Point objects and returns method-specific resampled points as a list of Point objects
        The resultant Point objects should be copies of the input Points BUT with Value overridden on each, e.g.:

        new_point = Point.copy(one_of_the_input_calibrated_points)
        for param in new_point.list_params():
          new_point.set_param_value(param, value=SOME_NEW_VALUE)

        :param calibrated_points: input points for this resampling method
        :return: a list of resampled Point objects
        """

        # selection_values: a DataFrame with columns relevant to selection of calibrated_points

        center_point = initial_calibration_points[0]

        # convert input points to DataFrames
        calibrated_points_df = []
        for i in range(len(calibrated_points)):
            new_item = calibrated_points[i].to_value_dict(parameter_type=CalibrationPoint.DYNAMIC)
            calibrated_points_df.append(new_item)
        calibrated_points_df = pd.DataFrame(calibrated_points_df)
        original_column_names = calibrated_points_df.columns
        calibrated_points_df = selection_values.join(calibrated_points_df)

        # ck4, debugging only, this block
        filename = os.path.join(self.output_location, 'cr-calibrated-points.csv')  # C
        calibrated_points_df.to_csv(filename)

        # temporary, generic column names for the actual parameter names
        # theta_column_names = ['theta%d'%i for i in range(len(original_column_names))]
        # temp_columns = list(selection_values.columns) + theta_column_names
        # calibrated_points_df.columns = temp_columns

        # same as calibrated_points_df but with a LL column on the end
        likelihood_df = pd.DataFrame([{'LL': point.likelihood} for point in calibrated_points])
        likelihood_df = calibrated_points_df.join(likelihood_df)

        # ck4, debugging only, this block
        filename = os.path.join(self.output_location, 'cr-calibrated-points-ll.csv') # E
        likelihood_df.to_csv(filename)

        # Do the resampling

        # center_point is a list of param values at the center point, must be ordered exactly as calibrated_points_df column-wise
        # obtain min/max of parameter ranges to force results to be within them
        minimums = center_point.get_attribute(key='Min', parameter_type=CalibrationPoint.DYNAMIC)
        maximums = center_point.get_attribute(key='Max', parameter_type=CalibrationPoint.DYNAMIC)
        names = center_point.get_attribute(key='Name', parameter_type=CalibrationPoint.DYNAMIC)

        center_point_as_list = list(pd.DataFrame([center_point.to_value_dict(parameter_type=CalibrationPoint.DYNAMIC)]).as_matrix()[0])

        fisher_inf_matrix = FisherInfMatrix(center_point_as_list, likelihood_df, names)
        covariance = np.linalg.inv(fisher_inf_matrix)

        ## the 2 principal components using PCA
        scaled_covariance = np.dot(np.dot(np.asmatrix(np.diag(1/(np.array(maximums)-np.array(minimums)))), covariance), np.asmatrix(np.diag(1/(np.array(maximums)-np.array(minimums)))))
        pca = PCA(n_components=2)
        pca.fit(scaled_covariance)
        print('covariance (scaled): \n', scaled_covariance)
        print('principal axes: \n', pca.components_)
        print('variance of first 2 principal components: \n', pca.explained_variance_)


        resampled_points_list = trunc_gauss(center_point_as_list, covariance, minimums, maximums,
                                            **self.resample_kwargs)

        # convert resampled points to a list of CalibrationPoint objects (multiple steps here)
        resampled_points_df = pd.DataFrame(data=resampled_points_list, columns=original_column_names)

        # attach static parameters
        names = center_point.get_attribute('Name', parameter_type=CalibrationPoint.STATIC)
        values = center_point.get_attribute('Value', parameter_type=CalibrationPoint.STATIC)
        for i in range(len(names)):
            resampled_points_df = resampled_points_df.assign(**{str(names[i]): values[i]})
        resampled_points_df.sort_index(axis=1, inplace=True)

        # ck4, debugging only
        filename = os.path.join(self.output_location, 'cr-resampled-points.csv') # J
        resampled_points_df.to_csv(filename)

        resampled_points = self._transform_df_points_to_calibrated_points(center_point,
                                                                          resampled_points_df)

        # ck4, debugging only
        from calibtool.resamplers.CalibrationPoints import CalibrationPoints
        filename = os.path.join(self.output_location, 'cr-resampled-points-transformed.csv') # K
        points = CalibrationPoints(resampled_points)
        points.write(filename)

        # this can be anything; will be made available for use in post_analysis() method in the from_resample argument
        for_post_analysis = None

        # pp = sns.pairplot(resampled_points_df[original_column_names], size=1.8, aspect=1.8,
        #              plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
        #
        # fig = pp.fig
        # fig.subplots_adjust(top=0.93, wspace=0.3)
        # fig.savefig(os.path.join(self.output_location, 'samples_pairs.pdf'))
        # fig.clf()
        # plt.close(fig)
        # del pp, fig




        # ck4, Verification, for review
        # J is the same as K, except for an additional indexing column in J
        # D is the same as E except for the LL column and end-of-line whitespace chars
        # E is the same as LLdata.csv from the RandomPerturbationResampler EXCEPT different param/LL column order (same values)
        #    -- This should be fine, as order preservation is only needed across (in/out) of CR resampling due to column renaming.
        # Parameter number ranges for input (cr-calibrated-points.csv) and output (cr-resampled-points.csv) are similar,
        #    e.g. col0 ~ 1.25, col1 ~ 2000, col2 ~ 0.65, col3 ~ 25

        # return reampled points
        return resampled_points, for_post_analysis


    def post_analysis(self, resampled_points, analyzer_results, from_resample=None):
        super().post_analysis(resampled_points, analyzer_results, from_resample=from_resample)
        output_filename = os.path.join(self.output_location, 'cr-resampled-points-ll.csv')
        resampled_points_ll_df = pd.DataFrame([rp.to_value_dict(include_likelihood=True) for rp in resampled_points])
        resampled_points_ll_df.to_csv(output_filename)

        """ plotting """
        resampled_points_dynamic = pd.DataFrame([rp.to_value_dict(parameter_type=CalibrationPoint.DYNAMIC, include_likelihood=True) for rp in resampled_points])


        pp = sns.pairplot(resampled_points_dynamic[resampled_points_dynamic.columns[0:-1]], size=1.8, aspect=1.8,
                     plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))

        fig = pp.fig
        fig.subplots_adjust(top=0.93, wspace=0.3)
        fig.savefig(os.path.join(self.output_location, 'samples_pairs.pdf'))
        fig.clf()
        plt.close(fig)
        del pp, fig

        ## just to test plot the first two dynamic parameters
        fig, ax = plt.subplots()
        x = resampled_points_dynamic[resampled_points_dynamic.columns[0]]
        y = resampled_points_dynamic[resampled_points_dynamic.columns[1]]
        colors = resampled_points_dynamic[resampled_points_dynamic.columns[-1]]
        plt.scatter(x, y, c=colors, alpha=0.3, cmap='viridis')
        plt.xlabel(resampled_points_dynamic.columns.values[0])
        plt.ylabel(resampled_points_dynamic.columns.values[1])
        plt.title('log-likelihood of the samples')
        plt.xlim([rp._filter_parameters(parameter_type=CalibrationPoint.DYNAMIC) for rp in resampled_points][0][0].min,
                 [rp._filter_parameters(parameter_type=CalibrationPoint.DYNAMIC) for rp in resampled_points][0][0].max)
        plt.ylim([rp._filter_parameters(parameter_type=CalibrationPoint.DYNAMIC) for rp in resampled_points][0][1].min,
                 [rp._filter_parameters(parameter_type=CalibrationPoint.DYNAMIC) for rp in resampled_points][0][1].max)
        plt.colorbar()
        plt.savefig(os.path.join(self.output_location, 'samples_pairs_LL.pdf'))
        fig.clf()
        plt.close(fig)
        del ax, fig

    def plot(self, center, Xmin, Xmax, df_perturbed_points, ll):
        """
        :param center:
        :param Xmin:
        :param Xmax:
        :param df_perturbed_points:
        :param ll: df_perturbed_points with likelihood
        :return:
        """
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Fisher = FisherInfMatrix(center, df_perturbed_points, ll)
        Covariance = np.linalg.inv(Fisher)

        print("eigs of fisher: ", np.linalg.eigvals(Fisher))
        print("eigs of Covariance: ", np.linalg.eigvals(Covariance))

        fig3 = plt.figure('CramerRao')
        ax = plt.subplot(111)
        x, y = center[0:2]
        plt.plot(x, y, 'g.')
        plot_cov_ellipse(Covariance[0:2, 0:2], center[0:2], nstd=3, alpha=0.6, color='green')
        plt.xlim(Xmin[0], Xmax[0])
        plt.ylim(Xmin[1], Xmax[1])
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)

        # save fig to file
        fig3.savefig(os.path.join(self.output_location, 'CramerRao.png'))

        plt.show()


