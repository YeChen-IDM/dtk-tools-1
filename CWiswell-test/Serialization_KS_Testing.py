from simtools.ModBuilder import SingleSimulationBuilder
from simtools.SetupParser import SetupParser
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from dtk.generic import serialization
from dtk.tools.serialization import combine_inset_charts
from dtk.tools.serialization import  ks_channel_testing

from os import path
import json

class SerializationKsTest(object):
    def __init__(self, inset_channels=[],
                 property_channels=[],
                 config_builder=None,
                 experiment_name='Serialization test',
                 experiment_tags={'role':'serialization_test'},
                 timesteps_to_serialize = []):
        self.inset_channels = inset_channels
        self.property_channels = property_channels
        self.cb = config_builder
        self.exp_name = experiment_name
        self.exp_tags = experiment_tags

        self.exp_manager = None
        self.baseline_sim = None
        self.timesteps_to_serialize = timesteps_to_serialize
        self.s_timestep_tagname = 'Serialization_Timestep'

    def run_test(self):
        SetupParser.init()
        self.exp_manager = ExperimentManagerFactory.init()
        exp_manager = self.exp_manager

        for tag in self.exp_tags:
            exp_manager.experiment_tags[tag] = self.exp_tags[tag]

        cfg_builder = self.cb
        cfg_builder.set_param("Config_Name", "Full run sim")
        fullrun_builder = SingleSimulationBuilder()
        fullrun_builder.tags['role'] = 'fullrun'
        exp_manager.run_simulations(config_builder=cfg_builder,
                                    exp_builder=fullrun_builder,
                                    exp_name=self.exp_name)

        old_sim_duration = cfg_builder.params["Simulation_Duration"]
        for ts in self.timesteps_to_serialize:
            ts_string = str(ts)
            serialization.add_SerializationTimesteps(config_builder=cfg_builder,
                                                     timesteps=[ts],
                                                     end_at_final=True)
            cfg_builder.set_param("Config_Name","Serializing sim at {0}".format(ts_string))
            serialized_builder = SingleSimulationBuilder()
            serialized_builder.tags['role'] = 'serializer'
            serialized_builder.tags[self.s_timestep_tagname] = ts_string
            exp_manager.run_simulations(config_builder=cfg_builder,
                                        exp_builder=serialized_builder,
                                        exp_name=self.exp_name)

        exp_manager.wait_for_finished()

        self.baseline_sim = exp_manager.experiment.get_simulations_with_tag('role','fullrun')[0]
        cfg_builder.params.pop("Serialization_Time_Steps")

        for ts in self.timesteps_to_serialize:
            ts_string = str(ts)
            serialized_sim = exp_manager.experiment.get_simulations_with_tag(self.s_timestep_tagname,ts_string)[0]
            serialized_output_path = path.join(serialized_sim.get_path(), 'output')

            # build s_pop_filename
            prefix_len = 5 - len(ts_string)
            prefix = '0' * prefix_len
            s_pop_filename = 'state-{0}{1}.dtk'.format(prefix, ts_string)

            cfg_builder.set_param("Config_Name","Reloading sim at {0}".format(ts_string))
            reloaded_builder = SingleSimulationBuilder()
            reloaded_builder.tags['role'] = 'reloader'
            reloaded_builder.tags[self.s_timestep_tagname] = ts_string
            cfg_builder.params["Start_Time"] = ts
            cfg_builder.params["Simulation_Duration"] = old_sim_duration - ts

            serialization.load_Serialized_Population(config_builder=cfg_builder,
                                                     population_filenames=[s_pop_filename],
                                                     population_path=serialized_output_path)
            exp_manager.run_simulations(config_builder=cfg_builder,
                                        exp_builder=reloaded_builder,
                                        exp_name=self.exp_name)
        exp_manager.wait_for_finished()
        self.create_combined_charts()
        if self.inset_channels:
            self.test_channels(self.inset_channels, original_report_name='InsetChart.json')

    def create_combined_charts(self):
        merged_insets = False
        merged_property_charts = False
        for ts in self.timesteps_to_serialize:
            ts_string = str(ts)
            sims_to_combine = self.exp_manager.experiment.get_simulations_with_tag(self.s_timestep_tagname,ts_string)
            s_sim = None
            r_sim = None
            print("sims to combine length: {0}\n".format(len(sims_to_combine)))
            print("sim[0] tags: {0}\n".format(sims_to_combine[0].tags))
            if 'serializer' == sims_to_combine[0].tags['role']:
                s_sim = sims_to_combine[0]
                r_sim = sims_to_combine[1]
            else:
                r_sim = sims_to_combine[0]
                s_sim = sims_to_combine[1]
            s_output_folder = path.join(s_sim.get_path(), 'output')
            r_output_folder = path.join(r_sim.get_path(), 'output')

            if self.inset_channels:
                merged_insets = True
                self.merge_json_charts(serialized_output_folder=s_output_folder,
                                       reloaded_output_folder=r_output_folder,
                                       original_report_name='InsetChart.json')
            if self.property_channels:
                merged_property_charts = True
                self.merge_json_charts(serialized_output_folder=s_output_folder,
                                       reloaded_output_folder=r_output_folder,
                                       original_report_name='PropertyReport.json')
        if merged_insets or merged_property_charts:
            baseline_output = path.join(self.baseline_sim.get_path(), 'output')
        if merged_insets: # Create a 'combined' report so you can look in COMPS
            self.merge_json_charts(serialized_output_folder=None,
                                   reloaded_output_folder=baseline_output,
                                   original_report_name='InsetChart.json')
        if merged_property_charts:
            self.merge_json_charts(serialized_output_folder=None,
                                   reloaded_output_folder=baseline_output,
                                   original_report_name='PropertyReport.json')

    def merge_json_charts(self, serialized_output_folder, reloaded_output_folder, original_report_name='InsetChart.json'):
        r_inset_chart = path.join(reloaded_output_folder, original_report_name)
        combined_chart_name = 'combined_{0}'.format(original_report_name)
        combined_chart_path = path.join(reloaded_output_folder, combined_chart_name)
        if serialized_output_folder:
            s_inset_chart = path.join(serialized_output_folder, original_report_name)
            combined_chart = combine_inset_charts.combine_channels(s_inset_chart, r_inset_chart)
        else:
            with open(r_inset_chart) as infile:
                combined_chart = json.load(infile)
            self.baseline_charts = {}
            self.baseline_charts[original_report_name] = r_inset_chart
        print ("combined chart path is {0}".format(combined_chart_path))
        with open(combined_chart_path, 'w') as outfile:
            json.dump(combined_chart, outfile, indent=4, sort_keys=True)

    def test_channels(self, channels_to_test, original_report_name="InsetChart.json"):
        for ts in self.timesteps_to_serialize:
            ts_string = str(ts)
            ts_sims = self.exp_manager.experiment.get_simulations_with_tag(self.s_timestep_tagname, ts_string)
            ts_reloaded = None
            for sim in ts_sims:
                if 'role' in sim.tags and 'reloader' == sim.tags['role']:
                    ts_reloaded = sim
                    break
            if not ts_reloaded:
                raise Exception("Couldn't find the reloader sim for timestep {0}".format(ts_string))
            test_output_folder = path.join(sim.get_path(), 'output')
            test_report_path = path.join(test_output_folder, original_report_name)
            KsTester = ks_channel_testing.KsChannelTester(
                ref_path=self.baseline_charts[original_report_name],
                test_path=test_report_path,
                channel_list=channels_to_test)
            new_tags = ts_reloaded.tags
            for c in channels_to_test:
                stat, p_val = KsTester.test_channel(c)
                tag_name = c.replace(' ','_')
                tag_name = tag_name.replace(':','_')
                tag_name = "step_{0}_{1}_{2}".format(ts_string, tag_name, 'PVal')
                new_tags[tag_name] = p_val
            ts_reloaded.set_tags(new_tags)
            ts_reloaded.description = "<img src=\"https://blog.arpinphilately.com/wp-content/uploads/2014-10-23_1428-128x128.png\" >"
            ts_reloaded.save()
        # get the reloaded sim
        # # for c in channels_to_test

