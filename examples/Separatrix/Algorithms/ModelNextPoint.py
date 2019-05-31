import json
import logging
import numpy as np
import pandas as pd
from calibtool.algorithms.NextPointAlgorithm import NextPointAlgorithm
from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection
from examples.Separatrix.Algorithms.AlgoHelper.igBDOE import igBDOE
from examples.Separatrix.Algorithms.AlgoHelper.tanhModel import tanhModel
from examples.Separatrix.Algorithms.AlgoHelper.SigmoidalModel import SigmoidalModel
from examples.Separatrix.Algorithms.AlgoHelper.utils import generate_requested_points, zeroCorners

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNextPoint(NextPointAlgorithm):
    """
    """

    def __init__(self, params=None, Settings={}, Num_Initial_Samples=10, Num_Next_Samples=10,
                 Num_Test_Points=10, Num_Candidates_Points=10, **kwargs):
        super().__init__()
        self.params = params
        self.Num_Dimensions = len(params)
        self.Num_Initial_Samples = Num_Initial_Samples
        self.Num_Next_Samples = Num_Next_Samples
        self.Num_Test_Points = Num_Test_Points
        self.Num_Candidates_Points = Num_Candidates_Points
        self.Settings = Settings
        self.data = pd.DataFrame()
        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)
        self.parameter_ranges = []
        self.test_points = pd.DataFrame()
        self.possible_points = pd.DataFrame()
        self.inference_x = None

        self.init(**kwargs)

    def init(self, **kwargs):
        # 1. build params ranges
        for p in self.params:
            min_v = p['Min']
            max_v = p['Max']
            self.parameter_ranges.append(dict(Min=min_v, Max=max_v))

        # 2. update counts as we will use Settings in code
        if self.Num_Dimensions is not None:
            self.Settings["Num_Dimensions"] = self.Num_Dimensions
        else:
            self.Num_Dimensions = self.Settings["Num_Dimensions"]

        if self.Num_Initial_Samples is not None:
            self.Settings["Num_Initial_Samples"] = self.Num_Initial_Samples
        else:
            self.Num_Initial_Samples = self.Settings["Num_Initial_Samples"]

        if self.Num_Next_Samples is not None:
            self.Settings["Num_Next_Samples"] = self.Num_Next_Samples
        else:
            self.Num_Next_Samples = self.Settings["Num_Next_Samples"]

        if self.Num_Test_Points is not None:
            self.Settings["Num_Test_Points"] = self.Num_Test_Points
        else:
            self.Num_Test_Points = self.Settings["Num_Test_Points"]

        if self.Num_Candidates_Points is not None:
            self.Settings["Num_Candidates_Points"] = self.Num_Candidates_Points
        else:
            self.Num_Candidates_Points = self.Settings["Num_Candidates_Points"]

        # 3. update possible extra parameters
        for key, value in kwargs.items():
            self.Settings[key] = value

        # 4. calculate inference_x
        if self.Num_Dimensions == 1:
            self.inference_x = np.linspace(0, self.parameter_ranges[0]['Max'],
                                           self.Settings["Inference_Grid_Resolution"])
            self.inference_x = self.inference_x.reshape(self.inference_x.shape[0], 1)
        elif self.Num_Dimensions == 2:
            ix, iy = np.meshgrid(
                np.linspace(0, self.parameter_ranges[0]['Max'], self.Settings["Inference_Grid_Resolution"]),
                np.linspace(0, self.parameter_ranges[1]['Max'], self.Settings["Inference_Grid_Resolution"]))
            self.inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T

        # 5. initialize model
        np.random.seed(self.Settings["Random_Seed"])
        myrng = np.random.rand()
        if self.Num_Dimensions == 1:
            self.model = SigmoidalModel(myrng=myrng)
        else:
            self.model = tanhModel(myrng=myrng)

    def get_test_sample_points(self, iteration):
        samples_all = self.test_points.copy()
        samples_all = samples_all[samples_all['Iteration'] == iteration]

        testPoints = self.convert_df_to_points(samples_all, include_results=False)
        return zeroCorners(testPoints)

    def get_possible_sample_points(self, iteration):
        samples_all = self.possible_points.copy()
        samples_all = samples_all[samples_all['Iteration'] == iteration]

        possibleSamplePoints = self.convert_df_to_points(samples_all, include_results=False)
        return zeroCorners(possibleSamplePoints)

    def choose_initial_samples(self):
        self.data = pd.DataFrame(
            columns=['Iteration', '__sample_index__', 'Results', *self.get_param_names()])
        self.data['Iteration'] = self.data['Iteration'].astype(int)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)
        self.data['Results'] = self.data['Results'].astype(int)

        iteration = 0

        # Clear self.state in case of resuming iteration 0 from commission
        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)

        for param in self.params:
            self.state.loc[len(self.state)] = [iteration, param['Name'], param['Min'], param['Max']]

        # Use LHS to generate points
        # points = self.get_lhs_samples(self.Settings["Num_Initial_Samples"])
        points = LHSPointSelection(self.Num_Initial_Samples, self.Num_Dimensions, ParameterRanges=self.parameter_ranges)
        initial_samples = self.convert_points_to_df(points)

        self.add_samples(initial_samples, iteration)

        # Generate test and possible points...
        self.post_samples(iteration=0)

        return initial_samples

    def choose_next_samples(self, iteration):
        """
        Shpuld construct from previous result
        :param iteration:
        :return:
        """
        # retrieve all previous samples
        sample_x, sample_y = self.get_all_samples()

        # [TODO]: Zdu: test (sample_y has nan results). Should be removed late!
        # sample_y = self.model.Sample(sample_x)

        # retrieve test and possible points
        testPoints = self.get_test_sample_points(iteration - 1)
        possibleSamplePoints = self.get_possible_sample_points(iteration - 1)

        # Generate the next samples
        new_sample_x, testPoints, possibleSamplePoints = igBDOE(sample_x, sample_y, self.inference_x,
                                                                self.parameter_ranges, self.Settings, testPoints,
                                                                possibleSamplePoints)

        # store new samples
        next_samples = self.convert_points_to_df(new_sample_x)
        self.add_samples(next_samples, iteration)

        # store new test samples
        testPoints = self.convert_points_to_df(testPoints)
        self.test_points = self.add_samples_to_df(testPoints, self.test_points, iteration)

        # store possible samples
        possibleSamplePoints = self.convert_points_to_df(possibleSamplePoints)
        self.possible_points = self.add_samples_to_df(possibleSamplePoints, self.possible_points, iteration)

        return next_samples

    # [TODO]: it may need to include test and possible points as well for resume!
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

    # Output test and possible points as well
    def get_final_samples_test(self):
        """
        Resample Stage:
        """
        iteration = self.data['Iteration'].max()
        data_by_iter = self.data[self.data['Iteration'] == iteration]
        final_samples = data_by_iter.drop(['Iteration', 'Results'], axis=1)

        data_by_iter = self.test_points[self.test_points['Iteration'] == iteration]
        test_samples = data_by_iter.drop(['Iteration'], axis=1)

        data_by_iter = self.possible_points[self.possible_points['Iteration'] == iteration]
        possible_samples = data_by_iter.drop(['Iteration'], axis=1)

        return {
            'final_samples': {
                'samples': final_samples.to_dict(orient='list'),
                'test_samples': test_samples.to_dict(orient='list'),
                'possible_samples': possible_samples.to_dict(orient='list')
            }
        }

    def get_sample_points(self, iteration):
        # Convert samples to matrix format
        samples_all = self.data.copy()
        samples_all = samples_all[samples_all['Iteration'] == iteration]

        return self.convert_df_to_points(samples_all)

    def convert_df_to_points(self, df, include_results=True):
        # Convert samples to matrix format
        samples = []
        if self.Num_Dimensions == 1:
            data_by_iter = df[['Point_X']]
            for index, rows in data_by_iter.iterrows():
                samples.append([rows.Point_X])
        elif self.Num_Dimensions == 2:
            data_by_iter = df[['Point_X', 'Point_Y']]
            for index, rows in data_by_iter.iterrows():
                samples.append([rows.Point_X, rows.Point_Y])

        sample_x = np.array(samples)

        if include_results:
            sample_y = np.array(df[['Results']])
            return sample_x, sample_y
        else:
            return sample_x

    def get_all_samples(self):
        # Convert samples to matrix format
        samples_all = self.data.copy()

        return self.convert_df_to_points(samples_all)

    def convert_points_to_df(self, points):
        points_df = pd.DataFrame(points)
        points_df.columns = [p["Name"] for p in self.params]

        return points_df

    def add_samples_to_df(self, samples, df, iteration):
        samples_cpy = samples.copy()
        samples_cpy.index.name = '__sample_index__'
        samples_cpy['Iteration'] = iteration
        samples_cpy.reset_index(inplace=True)

        df = pd.concat([df, samples_cpy], ignore_index=True)
        df['__sample_index__'] = df['__sample_index__'].astype(int)

        return df

    def post_samples(self, iteration):
        """
        Generate requested test points and possible interested points
        :param iteration:
        :return:
        """
        # generate and store test sample points
        test_points = generate_requested_points(self.Settings["Num_Test_Points"], self.Num_Dimensions,
                                                self.parameter_ranges)
        test_samples = self.convert_points_to_df(test_points)
        self.test_points = self.add_samples_to_df(test_samples, self.test_points, iteration)

        # generate and store possible sample points
        possible_points = generate_requested_points(self.Settings["Num_Candidates_Points"], self.Num_Dimensions,
                                                    self.parameter_ranges)
        possible_samples = self.convert_points_to_df(possible_points)
        self.possible_points = self.add_samples_to_df(possible_samples, self.possible_points, iteration)

    def get_state(self):
        return self.data

    def set_state(self, state, iteration):
        self.data = state

    def cleanup(self):
        pass

    def end_condition(self):
        return False


def test_2d():
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

    # Load Separatrix settings
    Settings = json.load(open('../Settings.json', 'r'))

    model_next_point = ModelNextPoint(params, Settings=Settings, Num_Initial_Samples=20,
                                      Num_Next_Samples=20)

    initial_samples = model_next_point.choose_initial_samples()
    print(initial_samples)

    next_samples = model_next_point.choose_next_samples(1)
    print(next_samples)

    next_samples = model_next_point.choose_next_samples(2)
    print(next_samples)


def test_1d():
    params = [
        {
            'Name': 'Point_X',
            'MapTo': 'Point_X',
            'Min': 0,
            'Max': 1
        }
    ]

    # Load Separatrix settings
    Settings = json.load(open('../Settings.json', 'r'))

    model_next_point = ModelNextPoint(params, Settings=Settings, Num_Initial_Samples=10,
                                      Num_Next_Samples=10)

    initial_samples = model_next_point.choose_initial_samples()
    print(initial_samples)

    next_samples = model_next_point.choose_next_samples(1)
    print(next_samples)

    next_samples = model_next_point.choose_next_samples(2)
    print(next_samples)


if __name__ == "__main__":
    test_2d()
    exit()

    test_1d()
    exit()
