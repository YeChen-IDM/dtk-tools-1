from dtk.utils.analyzers.DownloadAnalyzerTPI import DownloadAnalyzerTPI
from simtools.AnalyzeManager.AnalyzeManager import AnalyzeManager
from simtools.SetupParser import SetupParser
from simtools.Utilities.Experiments import retrieve_experiment

if __name__ == "__main__":
    SetupParser.init('HPC')
    experiment = retrieve_experiment('8e35aced-d470-e711-9401-f0921c16849d') # '5d35c13c-986d-e711-9401-f0921c16849d')
    am = AnalyzeManager(exp_list=experiment)
    am.add_analyzer(DownloadAnalyzerTPI(filenames=['config.json', 'output\\InsetChart.json']))
    am.analyze()
