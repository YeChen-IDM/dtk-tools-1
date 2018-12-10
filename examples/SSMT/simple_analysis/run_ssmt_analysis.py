from examples.SSMT.simple_analysis.analyzers.AdultVectorsAnalyzer import AdultVectorsAnalyzer
from examples.SSMT.simple_analysis.analyzers.PopulationAnalyzer import PopulationAnalyzer
from simtools.Analysis.SSMTAnalysis import SSMTAnalysis
from simtools.SetupParser import SetupParser

if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()

    analysis = SSMTAnalysis(experiment_ids=["d06218be-53d2-e811-80ca-f0921c167866"],
                            analyzers=[PopulationAnalyzer(), AdultVectorsAnalyzer()],
                            analyzers_args=[{'title': 'iv'}, {'name': 'global good'}],
                            analysis_name="SSMT Analysis 1")

    analysis.analyze()
