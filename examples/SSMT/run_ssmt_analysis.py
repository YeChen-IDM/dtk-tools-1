from examples.SSMT.analyzers.AdultVectorsAnalyzer import AdultVectorsAnalyzer
from examples.SSMT.analyzers.PopulationAnalyzer import PopulationAnalyzer
from simtools.Analysis.SSMTAnalysis import SSMTAnalysis
from simtools.SetupParser import SetupParser

if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()

    analysis = SSMTAnalysis(experiment_ids=["4c019e95-43e4-e811-80ca-f0921c167866"],
                            analyzers=[PopulationAnalyzer(), AdultVectorsAnalyzer()],
                            analysis_name="SSMT Analysis 1")

    analysis.analyze()
