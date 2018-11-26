from examples.SSMT.simple_analysis.analyzers.AdultVectorsAnalyzer import AdultVectorsAnalyzer
from examples.SSMT.simple_analysis.analyzers.PopulationAnalyzer import PopulationAnalyzer
from simtools.Analysis.SSMTAnalysis import SSMTAnalysis
from simtools.SetupParser import SetupParser

if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()

    analysis = SSMTAnalysis(experiment_ids=["39953ccf-e899-e811-a2c0-c4346bcb7275"],
                            analyzers=[PopulationAnalyzer(), AdultVectorsAnalyzer()],
                            analysis_name="SSMT Analysis 1")

    analysis.analyze()
