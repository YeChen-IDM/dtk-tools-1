import os

from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate
from simtools.SetupParser import SetupParser

SetupParser.default_block = 'HPC'

# Load the base files
plugin_files_dir = 'SamplesInput'
config = ConfigTemplate.from_file(os.path.join(plugin_files_dir, 'config.json'))
cpn = CampaignTemplate.from_file(os.path.join(plugin_files_dir, 'campaign.json'), '__KP')

# Load the scenarios
headers = [
    "Name",
    "p1",
    "p2"
]
scenarios = [
    ['Scenario 1', ]
]