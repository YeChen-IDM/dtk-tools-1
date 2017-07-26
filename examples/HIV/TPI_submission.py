import os
import time

import pandas

from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate
from dtk.utils.builders.TemplateHelper import TemplateHelper
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.ModBuilder import ModBuilder
from simtools.SetupParser import SetupParser

SetupParser.default_block = 'HPC'


def header_table_to_dict(header, table):
    df = pandas.DataFrame(data=table, columns=header)
    return df.to_dict(orient='records')

# Load the base files
plugin_files_dir = 'Templates'
config = ConfigTemplate.from_file(os.path.join(plugin_files_dir, 'config.json'))
cpn = CampaignTemplate.from_file(os.path.join(plugin_files_dir, 'campaign.json'), '__KP')

# Create the default config builder
config_builder = DTKConfigBuilder.from_files(
    os.path.join(plugin_files_dir, 'config.json'),
    os.path.join(plugin_files_dir, 'campaign.json')
 )

# Load the scenarios
s_headers = [
    "Name",
    "Start_Year__KP_Seeding_Year",
    "Condom_Usage_Probability__KP_INFORMAL.Max"
]
scenarios = [
    ['Scenario 1', 1980, 0.123],
    ['Scenario 2', 1981, 0.456]
]

# And the points
p_headers = [
    "Run_Number",
    "Base_Infectivity"
]
points = [
    [1, 0.5],
    [2, 0.6]
]

# Get the dicts
points_dict = header_table_to_dict(p_headers, points)
scenarios_dict = header_table_to_dict(s_headers, scenarios)

# Experiments containing all the scenarios
experiments = []

if __name__ == "__main__":
    SetupParser.init()
    # Create the scenarios
    for scenario in scenarios_dict:
        scenario_name = scenario.pop('Name')
        # For each scenario, combine with the points first
        combined = []
        for point in points_dict:
            current = {}
            current.update(scenario)
            current.update(point)
            combined.append(current)

        # Extract the headers
        headers = combined[0].keys()

        # Construct the table
        table = [c.values() for c in combined]

        # Initialize the template
        tpl = TemplateHelper()
        tpl.set_dynamic_header_table(headers, table)
        tpl.active_templates = [config, cpn]

        # Create an experiment builder
        experiment_builder = ModBuilder.from_combos(tpl.get_modifier_functions())
        experiment_manager = ExperimentManagerFactory.from_cb(config_builder)
        experiment_manager.bypass_missing = True
        experiment_manager.run_simulations(exp_name=scenario_name, exp_builder=experiment_builder)
        experiments.append(experiment_manager)

    finished = False
    while not finished:
        finished = True
        for em in experiments:
            if not em.finished():
                finished = False
                states, msg = em.get_simulation_status()
                em.print_status(states,msg)
                em.refresh_experiment()

        if not finished: time.sleep(3)




