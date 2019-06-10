import io
import json
import numpy as np
import pandas as pd
from simtools.Analysis.BaseAnalyzers import BaseCalibrationAnalyzer


class ModelAnalyzer(BaseCalibrationAnalyzer):

    def __init__(self, reference_data):
        super().__init__(filenames=['output\\result.json'], reference_data=reference_data, parse=False)

    def select_simulation_data(self, sim_data, simulation):
        data = json.load(io.BytesIO(sim_data[self.filenames[0]]))

        # Calculate the interested properties for comparison with the reference data
        result_dict = data.copy()
        result_dict["sample_index"] = data['__sample_index__']
        result_dict.pop('__sample_index__')

        # Returns the data needed for this simulation
        return result_dict

    def finalize(self, all_data):
        data = sorted(all_data.values(), key=lambda k: k['sample_index'])

        df = pd.DataFrame(data)

        # Here is a chance to do something with the data, say calculate LL or plotting
        # self.plot_demo(df)

        # Note: More information about plotting:
        # Here all_data is the results of all simulation in current iteration. As Separatrix' plotting needs simulation
        # results from the previous iteration, so we can't do the plotting here and we will do the plotting with plotter

        return df['Result']

    def plot_demo(self, df):
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # Make an array of x values
        x = df['sample_index'].tolist()
        # Make an array of y values for each x value
        y = df['Point_X'].tolist()

        ax1.plot(x, y)

        ax1.set_xticks(np.arange(len(x)))

        ax1.set_xlabel('sample_index')
        ax1.set_ylabel('Clinical Fever Threshold High')

        plt.title('Plot Demo', fontproperties=font_set)

        # show the plot on the screen
        plt.show()

        # fig.savefig('demo.pdf')
        # fig.savefig('demo.png')

        plt.close(fig)
