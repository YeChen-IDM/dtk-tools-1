import json
import os
import time

import pandas

from dtk.utils.analyzers.DownloadAnalyzerTPI import DownloadAnalyzerTPI
from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate, DemographicsTemplate
from dtk.utils.builders.TemplateHelper import TemplateHelper
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.AnalyzeManager.AnalyzeManager import AnalyzeManager
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.ModBuilder import ModBuilder
from simtools.SetupParser import SetupParser
from simtools.Utilities.Matlab.matreader import read_mat_points_file

SetupParser.default_block = 'HPC'
tpi_matlab_filename = 'MAT_file_containing_calibration_table/ReSample64.mat'
resume = False
# resume = ['6ec447cd-0f77-e711-9401-f0921c16849d','58c447cd-0f77-e711-9401-f0921c16849d']


# Unused parameters to remove
unused_params = [
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.MEDIUM.Prob_Extra_Relationship_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.LOW.Max_Simultaneous_Relationships_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.MARITAL.Concurrency_Parameters.MEDIUM.Max_Simultaneous_Relationships_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.LOW.Prob_Extra_Relationship_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.MEDIUM.Max_Simultaneous_Relationships_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.LOW.Max_Simultaneous_Relationships_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.MARITAL.Concurrency_Parameters.MEDIUM.Max_Simultaneous_Relationships_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.LOW.Prob_Extra_Relationship_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.MEDIUM.Prob_Extra_Relationship_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.MEDIUM.Max_Simultaneous_Relationships_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.MEDIUM.Max_Simultaneous_Relationships_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.MEDIUM.Prob_Extra_Relationship_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.LOW.Max_Simultaneous_Relationships_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.LOW.Prob_Extra_Relationship_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.LOW.Prob_Extra_Relationship_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.MEDIUM.Prob_Extra_Relationship_Female',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.TRANSITORY.Concurrency_Parameters.MEDIUM.Max_Simultaneous_Relationships_Male',
 'DEMOGRAPHICS.Society__KP_Defaults_All_Nodes.INFORMAL.Concurrency_Parameters.LOW.Max_Simultaneous_Relationships_Male'
]


def header_table_to_dict(header, table, index_name=None):
    df = pandas.DataFrame(data=table, columns=header)
    for unused in unused_params:
        if unused in df.columns: df.drop(unused,1, inplace=True)
    if index_name:
        df[index_name] = df.index

    return json.loads(pandas.json.dumps(df.to_dict(orient='records')))

# Load the base files
current_dir = os.path.dirname(os.path.realpath(__file__))
plugin_files_dir = os.path.join(current_dir,'EMOD_INPUT_FILES_templates')

# Load the base config file
config = ConfigTemplate.from_file(os.path.join(plugin_files_dir, 'Configs', 'config.json'))
config.set_param("Enable_Demographics_Builtin", 0, allow_new_parameters=True)

# Load the campaigns
cpn = CampaignTemplate.from_file(os.path.join(plugin_files_dir, 'Campaigns', 'campaign_Nyanza_Baseline_StatusQuo.json'))
cpnFT = CampaignTemplate.from_file(os.path.join(plugin_files_dir, 'Campaigns', 'campaign_Nyanza_ConAdh_FT.json'))
cpnSQ = CampaignTemplate.from_file(os.path.join(plugin_files_dir, 'Campaigns', 'campaign_Nyanza_ConAdh_SQ.json'))
campaigns = {"cpnFT":cpnFT, "cpnSQ":cpnSQ}

# Load the demographics
demog = DemographicsTemplate.from_file( os.path.join(plugin_files_dir, 'Demographics','Demographics.json'))
demog_pfa = DemographicsTemplate.from_file( os.path.join(plugin_files_dir,'Demographics' ,'PFA_Overlay.json'))
demog_acc = DemographicsTemplate.from_file( os.path.join(plugin_files_dir, 'Demographics','Accessibility_and_Risk_IP_Overlay.json'))
demog_asrt = DemographicsTemplate.from_file( os.path.join(plugin_files_dir,'Demographics','Risk_Assortivity_Overlay.json'))

# Load the scenarios
scenario_header = [
    'Start_Year__KP_PrEP_HIGH_Start_Year',
    'Event_Coordinator_Config__KP_PrEP_HIGH_Event.Time_Value_Map.Values',
    'Event_Coordinator_Config__KP_PrEP_HIGH_Event.Target_Gender',
    'Event_Coordinator_Config__KP_PrEP_HIGH_Event.Target_Age_Min',
    'Event_Coordinator_Config__KP_PrEP_HIGH_Event.Target_Age_Max',
    'Waning_Config__KP_PrEP_HIGH_Waning.Durability_Map.Times',
    'Waning_Config__KP_PrEP_HIGH_Waning.Initial_Effect',
    'Start_Year__KP_PrEP_MED_Start_Year',
    'Event_Coordinator_Config__KP_PrEP_MED_Event.Time_Value_Map.Values',
    'Event_Coordinator_Config__KP_PrEP_MED_Event.Target_Gender',
    'Event_Coordinator_Config__KP_PrEP_MED_Event.Target_Age_Min',
    'Event_Coordinator_Config__KP_PrEP_MED_Event.Target_Age_Max',
    'Waning_Config__KP_PrEP_MED_Waning.Durability_Map.Times',
    'Waning_Config__KP_PrEP_MED_Waning.Initial_Effect',
    'Campaign_Template',
    'Scenario'
]

scenarios = [
    [2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_Baseline'],
    [2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H40_M00_15_20'],
    [2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H40_M00_20_25'],
    [2017, [0, 0.9], 'Female', 15, 20, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H90_M00_15_20'],
    [2017, [0, 0.9], 'Female', 20, 25, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H90_M00_20_25'],
    [2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H40_M40_15_20'],
    [2017, [0, 0.9], 'Female', 15, 20, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H90_M40_15_20'],
    [2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H40_M40_20_25'],
    [2017, [0, 0.9], 'Female', 20, 25, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, "cpnSQ", 'StatusQuo_PrEP_F_H90_M40_20_25'],
    [2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_Baseline'],
    [2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H40_M00_15_20'],
    [2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H40_M00_20_25'],
    [2017, [0, 0.9], 'Female', 15, 20, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H90_M00_15_20'],
    [2017, [0, 0.9], 'Female', 20, 25, [0, 5 * 365], 0.73, 2099, [0, 1.0], 'Female', 0, 1, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H90_M00_20_25'],
    [2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H40_M40_15_20'],
    [2017, [0, 0.9], 'Female', 15, 20, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 15, 20, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H90_M40_15_20'],
    [2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H40_M40_20_25'],
    [2017, [0, 0.9], 'Female', 20, 25, [0, 5 * 365], 0.73, 2017, [0, 0.4], 'Female', 20, 25, [0, 5 * 365], 0.73, "cpnFT", 'FastTrack_PrEP_F_H90_M40_20_25']
]

# And the points
point_header, points = read_mat_points_file(tpi_matlab_filename)

# Create the default config builder
config_builder = DTKConfigBuilder()

# Set which executable we want to use for the experiments in the script
config_builder.set_experiment_executable('Eradication_Memory_Plus_GH826.exe')
# This is REQUIRED by the templates
config_builder.ignore_missing = True


# Get the dicts
points_dict = header_table_to_dict(point_header, points, index_name='TPI')
for point in points_dict:
    tpi = point.pop('TPI')
    if not 'TAGS' in point:
        point['TAGS'] = {}

    point['TAGS']['TPI'] = tpi

scenarios_dict = header_table_to_dict(scenario_header, scenarios)

if __name__ == "__main__":
    SetupParser.init()

    # Experiments containing all the scenarios
    experiments = []

    # Create the scenarios
    for scenario in scenarios_dict:
        if resume: break
        scenario_name = scenario.pop('Scenario')
        campaign_tpl = campaigns[scenario.pop('Campaign_Template')]

        # For each scenario, combine with the points first
        combined = []
        for point in points_dict:
            current = {}
            current.update(scenario)
            current.update(point)
            combined.append(current)

        # Extract the headers
        headers = [k.replace('CONFIG.', '').replace('DEMOGRAPHICS.', '').replace('CAMPAIGN.', '') for k in combined[0].keys()]

        # Construct the table
        table = [c.values() for c in combined]

        # Initialize the template
        tpl = TemplateHelper()
        tpl.set_dynamic_header_table(headers, table)
        tpl.active_templates = [config, cpn, campaign_tpl, demog, demog_pfa, demog_asrt, demog_acc]

        # Create an experiment builder
        experiment_builder = ModBuilder.from_combos(tpl.get_modifier_functions())
        experiment_manager = ExperimentManagerFactory.from_cb(config_builder)
        experiment_manager.bypass_missing = True
        experiment_manager.run_simulations(exp_name=scenario_name, exp_builder=experiment_builder)
        experiments.append(experiment_manager)

    if resume:
        experiments = [ExperimentManagerFactory.from_experiment(e) for e in resume]

    am = AnalyzeManager(verbose=False, create_dir_map=False)
    for em in experiments:
        am.add_experiment(em.experiment)
    am.add_analyzer(DownloadAnalyzerTPI(['output\\InsetChart.json']))

    while not all([em.finished() for em in experiments]):
        map(lambda e: e.refresh_experiment(), experiments)
        print("Analyzing !")
        am.analyze()
        print("Waiting 15 seconds")
        time.sleep(15)

    am.analyze()