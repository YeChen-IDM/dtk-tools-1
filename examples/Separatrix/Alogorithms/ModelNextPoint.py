import logging
import numpy as np
import pandas as pd
from calibtool.algorithms.GenericIterativeNextPoint import GenericIterativeNextPoint
from calibtool.algorithms.NextPointAlgorithm import NextPointAlgorithm

import lhsmdu
import matplotlib.pyplot as plt
from examples.Separatrix.Alogorithms.AlgoHelper.LHS import LHSPointSelection
from examples.Separatrix.Alogorithms.AlgoHelper.igBDOE import igBDOE
from examples.Separatrix.Alogorithms.AlgoHelper.tanhModel import tanhModel

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNextPoint(GenericIterativeNextPoint):
    """
    """

    def __init__(self, params=None, Num_Initial_Samples=15, Num_Next_Samples=15, Num_Test_Points=10,
                 Inference_Grid_Resolution=10, Interest_Level=0.7, Scale_Model_Sigma=0.02):
        super().__init__(None)
        self.params = params
        self.Num_Initial_Samples = Num_Initial_Samples
        self.Num_Next_Samples = Num_Next_Samples
        self.Num_Test_Points = Num_Test_Points
        self.Inference_Grid_Resolution = Inference_Grid_Resolution
        self.Interest_Level = Interest_Level
        self.Scale_Model_Sigma = Scale_Model_Sigma
        self.data = pd.DataFrame()
        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)

    def choose_initial_samples_bk(self):
        self.data = pd.DataFrame(
            columns=['Iteration', '__sample_index__', 'Results', *self.get_param_names()])
        self.data['Iteration'] = self.data['Iteration'].astype(int)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)

        iteration = 0

        # Clear self.state in case of resuming iteration 0 from commission
        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)

        for param in self.params:
            self.state.loc[len(self.state)] = [iteration, param['Name'], param['Min'], param['Max']]

        initial_samples = pd.DataFrame()
        for p in self.params:
            col_name = p['Name']
            min_v = p['Min']
            max_v = p['Max']
            type_v = p['Type']
            initial_samples[col_name] = [np.random.uniform(min_v, max_v) for _ in range(self.Num_Initial_Samples)]
            initial_samples[col_name] = initial_samples[col_name].astype(type_v)

        self.add_samples(initial_samples, iteration)

        return initial_samples

    def choose_initial_samples(self):
        self.data = pd.DataFrame(
            columns=['Iteration', '__sample_index__', 'Results', *self.get_param_names()])
        self.data['Iteration'] = self.data['Iteration'].astype(int)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)
        self.data['Results'] = self.data['Results'].astype(int)

        iteration = 0
        parameter_ranges = []

        # Clear self.state in case of resuming iteration 0 from commission
        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)

        for param in self.params:
            self.state.loc[len(self.state)] = [iteration, param['Name'], param['Min'], param['Max']]
            parameter_ranges.append(dict(Min=param['Min'], Max=param['Max']))

        points = LHSPointSelection(self.Num_Initial_Samples, ParameterRanges=parameter_ranges)

        initial_samples = pd.DataFrame(points)
        initial_samples.columns = [p["Name"] for p in self.params]

        self.add_samples(initial_samples, iteration)

        return initial_samples

    def choose_next_samples(self, iteration):
        """
        Shpuld construct from previous result
        :param iteration:
        :return:
        """

        grid_res = self.Inference_Grid_Resolution

        np.random.seed(1)
        myrng = np.random.rand()
        model = tanhModel(myrng=myrng)

        samples_all = self.data[['Point_X', 'Point_Y']]
        samples = []
        for index, rows in samples_all.iterrows():
            samples.append([rows.Point_X, rows.Point_Y])

        sample_x = np.array(samples)
        if 'Result' in self.data.columns.tolist():
            sample_y = self.data['Results']
        else:
            # Compute true/false outputs at the initial sample points
            sample_y = model.Sample(sample_x)

        # test purpose: self.data['Results'] is not populated yet!
        if np.isnan(sample_y[0]):
            sample_y = model.Sample(sample_x)

        parameter_ranges = []
        for param in self.params:
            parameter_ranges.append(dict(Min=param['Min'], Max=param['Max']))

        ix, iy = np.meshgrid(np.linspace(0, parameter_ranges[0]['Max'], grid_res), np.linspace(0, parameter_ranges[1]['Max'], grid_res))
        inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T
        # print(inference_x)

        new_sample_x = igBDOE(sample_x, sample_y, inference_x, parameter_ranges, self.params)

        # do it in ModelProcessor
        # new_sample_y = model.Sample(new_sample_x)

        # Join new samples with old
        # next_samples_x = np.concatenate((sample_x, new_sample_x), axis=0)
        # next_samples_y = np.concatenate((sample_y, new_sample_y), axis=0)

        next_samples = pd.DataFrame(new_sample_x)
        next_samples.columns = [p["Name"] for p in self.params]

        self.add_samples(next_samples, iteration)

        return next_samples

    def get_samples_for_iteration(self, iteration):
        if iteration == 0:
            samples = self.choose_initial_samples()
        else:
            samples = self.choose_next_samples(iteration)

        samples.reset_index(drop=True, inplace=True)
        return self.generate_samples_from_df(samples)

    def set_results_for_iteration(self, iteration, results):
        logger.info('%s: Choosing samples at iteration %d:', self.__class__.__name__, iteration)
        tf_col_name = results.columns.tolist()[0]

        results = results[tf_col_name].tolist()

        data_by_iter = self.data.set_index('Iteration')
        if iteration + 1 in data_by_iter.index.unique():
            # Been here before, reset
            data_by_iter = data_by_iter.loc[:iteration]

            state_by_iter = self.state.set_index('Iteration')
            self.state = state_by_iter.loc[:iteration].reset_index()

        # Store results ... even if changed
        print('set_results_for_iteration...')
        print(data_by_iter)
        print('-------')
        print(results)
        data_by_iter.loc[iteration, 'Results'] = results
        self.data = data_by_iter.reset_index()

        self.data['Results'] = self.data['Results'].astype(int)

    def get_param_names(self):
        return [p['Name'] for p in self.params]

    def add_samples(self, samples, iteration):
        samples_cpy = samples.copy()
        samples_cpy.index.name = '__sample_index__'
        samples_cpy['Iteration'] = iteration
        samples_cpy.reset_index(inplace=True)

        self.data = pd.concat([self.data, samples_cpy], ignore_index=True)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)

    def update_summary_table(self, iteration_state, previous_results):
        data = self.data.copy()
        data.rename(columns={'Iteration': 'iteration'}, inplace=True)
        return data, data

    def get_final_samples(self):
        """
        Resample Stage:
        """
        iteration = self.data['Iteration'].max()
        data_by_iter = self.data[self.data['Iteration'] == iteration]
        final_samples = data_by_iter.drop(['Iteration', 'Results'], axis=1)

        return {'final_samples': final_samples.to_dict(orient='list')}


if __name__ == "__main__":
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

    model_next_point = ModelNextPoint(params, Num_Initial_Samples=5, Num_Next_Samples=5, Num_Test_Points=6,
                                      Interest_Level=0.7, Scale_Model_Sigma=0.02)

    initial_samples = model_next_point.choose_initial_samples()
    print(initial_samples)

    next_samples = model_next_point.choose_next_samples(1)
    print(next_samples)

    next_samples = model_next_point.choose_next_samples(1)
    print(next_samples)