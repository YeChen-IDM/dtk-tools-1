import os
import sys
import logging
import re
import errno

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dtk.utils.analyzers.BaseShelveAnalyzer import BaseShelveAnalyzer

#logger = logging.getLogger(__name__)

class PostProcessAnalyzer(BaseShelveAnalyzer):
    prevalent_sheets = ['Population', 'Infected', 'ANC_Infected', 'ANC_Population', 'Number_On_ART']

    def __init__(self,
                max_sims_per_scenario = -1,
                compute_pop_scaling = True,
                reference_year = 2012.5,
                reference_population = 6269676,
                age_min = 15,
                age_max = 50,
                start_year = 2017, # Useful for analyzing interventions
                node_map = {},      # To name nodes
                node_order = None,
                intervention_subset = None,
                force_apply = False,
                force_combine = True,
                basedir = '.',
                fig_format = 'png',
                fig_dpi = 600,
                verbose = False,
                debug = False,
                **kwargs):

        self.ps_ave = {'Count': 0}
        self.count = {}
        self.skipped_interventions = []

        super(PostProcessAnalyzer, self).__init__(force_apply, force_combine, verbose)

        #self.parse = False   # Do not have dtk-tools parse the file, just provide it raw
        self.parse = True
        self.compute_pop_scaling = compute_pop_scaling

        if self.compute_pop_scaling:
            self.filenames = [ os.path.join('output', 'post_process', 'PopScaling.csv') ]

        # For pop scaling - would rather get from PopulationScalingAnalyzer!
        self.reference_year = reference_year
        self.reference_population = reference_population
        self.age_min = age_min
        self.age_max = age_max

        # Set to -1 to process all:
        self.max_sims_per_scenario = max_sims_per_scenario

        # Set to None to process all interventions, or pass a list of intervention names
        self.intervention_subset = intervention_subset

        self.start_year = start_year

        # Map from NodeId to Name - would rather get from a NodeListAnalyzer that reads Demographics.json for one simulation
        self.node_map = node_map
        self.node_order = node_order    # Can be None

        self.basedir = basedir
        self.fig_format = fig_format
        self.fig_dpi = fig_dpi
        self.verbose = verbose
        self.debug = debug

        #logger.info(self.__class__.__name__ + " writing to " + self.basedir) # TODO

        self.sim_ids = []
        self.num_outstanding = 0

        if not os.path.isdir(self.basedir):
            PostProcessAnalyzer.mkdir_p(self.basedir)

    @staticmethod
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    def filter(self, sim_metadata):
        self.workdir = os.path.join(self.basedir, self.exp_id)
        if not os.path.isdir(self.workdir):
            PostProcessAnalyzer.mkdir_p(self.workdir)

        self.figdir = os.path.join(self.workdir, self.__class__.__name__)
        if not os.path.isdir(self.figdir):
            PostProcessAnalyzer.mkdir_p(self.figdir)

        scenario = 'N/A'
        if 'Scenario' in sim_metadata:
            scenario = sim_metadata['Scenario']

        # SELECT BY SCENARIO ##################################################
        if self.intervention_subset is not None and self.intervention_subset:
            # TODO:
            if '_' in scenario:
                tok = scenario.split('_')
                intervention = '_'.join(tok[1:])

                if intervention not in self.intervention_subset:
                    if intervention not in self.skipped_interventions:
                        self.skipped_interventions.append( intervention )
                    if self.verbose:
                        print('Skipping', scenario)
                    return False
        #######################################################################

        # SELECT A LIMITED NUMBER #############################################
        if scenario not in self.count:
            self.count[scenario] = 1
        else:
            self.count[scenario] += 1

        if self.max_sims_per_scenario > 0 and self.count[scenario] > self.max_sims_per_scenario: # Take only a few of each scenario
            return False
        #######################################################################

        sim_id = sim_metadata['sim_id']
        self.sim_ids.append(sim_id)
        self.num_outstanding += 1

        if not self.shelve_file:    # Want this in the base class, but don't know exp_id at __init__
            self.shelve_file = os.path.join(self.workdir, '%s.db' % self.__class__.__name__) # USE ID instead?

        ret = super(PostProcessAnalyzer, self).filter(self.shelve_file, sim_metadata)

        if not ret and self.verbose:
            self.num_outstanding -= 1
            print('Skipping simulation %s because already in shelve' % str(sim_id))

        return ret

    @staticmethod
    def lower_value(agebin): # For age-bin sorting
        return [int(s) for s in re.findall(r'\b\d+\b', agebin)][0]

    @staticmethod
    def upper_value(agebin): # For age-bin sorting
        return [int(s) for s in re.findall(r'\b\d+\b', agebin)][1]

    def apply(self, parser, add_both_level_to_gender=True):
        super(PostProcessAnalyzer, self).apply(parser)

        self.num_outstanding -= 1
        if not self.compute_pop_scaling:
            if self.verbose:
                print('Progress: %d of %d (%.1f%%).'%(len(self.sim_ids)-self.num_outstanding, len(self.sim_ids), 100*(len(self.sim_ids)-self.num_outstanding) / float(len(self.sim_ids))))
            return


        # TODO: next(x for x in seq if predicate(x)); # It raises StopIteration if there is none.
        pop_scaling_idx = -1
        for fi, f in enumerate(self.filenames):
            ret = f.find('PopScaling.csv')
            if ret >= 0:
                pop_scaling_idx = fi
                break
        assert( pop_scaling_idx >= 0)

        #from StringIO import StringIO
        #csv_str = StringIO(parser.raw_data[self.filenames[pop_scaling_idx]])

        #parser.raw_data[self.filenames[pop_scaling_idx]].seek(0)
        #csv_str = parser.raw_data[self.filenames[pop_scaling_idx]]

        #pop = pd.read_csv(csv_str)

        pop = parser.raw_data[self.filenames[pop_scaling_idx]]

        ### POP SCALING #######################################################
        assert( self.reference_year in pop['Year'].unique() )
        assert( self.age_min in [PostProcessAnalyzer.lower_value(ab) for ab in pop['AgeBin'].unique()] )
        assert( self.age_max in [PostProcessAnalyzer.upper_value(ab) for ab in pop['AgeBin'].unique()] )

        if isinstance(self.reference_population, float) or isinstance(self.reference_population, int):
            sim_pop = pop.set_index('Year').loc[self.reference_year]['Result'].sum()
            pop_scaling = self.reference_population / float(sim_pop)
            if self.verbose:
                print('Progress: %d of %d (%.1f%%).  Pop scaling is %.2f.'%(len(self.sim_ids)-self.num_outstanding, len(self.sim_ids), 100*(len(self.sim_ids)-self.num_outstanding) / float(len(self.sim_ids)), pop_scaling))
        else:
            assert( isinstance(self.reference_population, dict) )
            pop = pop.reset_index(drop=True).set_index('Node')
            pop.rename(self.node_map, inplace=True)
            pop = pop.reset_index().groupby(['Year', 'Node']).sum().loc[self.reference_year]
            ref = pd.DataFrame({'Reference': self.reference_population})
            pop_scaling = ref['Reference']/pop['Result']
            if self.verbose:
                print('Progress: %d of %d (%.1f%%).  Pop scaling is provincially weighted from %.2f to %.2f.'%(len(self.sim_ids)-self.num_outstanding, len(self.sim_ids), 100*(len(self.sim_ids)-self.num_outstanding) / float(len(self.sim_ids)), pop_scaling.min(), pop_scaling.max()))

            for prov, pop in pop_scaling.items():
                if prov in self.ps_ave:
                    self.ps_ave[prov] += pop
                else:
                    self.ps_ave[prov] = pop
            self.ps_ave['Count'] += 1
        #######################################################################

        return {'Pop_Scaling':pop_scaling}


    def combine(self, parsers):

        if self.ps_ave['Count'] > 0:
            print('Pop scaling average:')
            c = self.ps_ave.pop('Count')
            for prov, pop in self.ps_ave.items():
                print(prov, ':', pop / float(c))

        if self.verbose:
            print("combine")

        if self.verbose and self.skipped_interventions:
            print('*' * 80)
            print('The following interventions were skipped:')
            print(self.skipped_interventions)
            print('*' * 80)

        return super(PostProcessAnalyzer, self).combine(parsers)

    def finalize(self):
        if self.verbose:
            print("finalize")

        super(PostProcessAnalyzer, self).finalize() # Closes the shelve file
        self.sim_ids = []

        sns.set_style("whitegrid")

    def plot(self):
        plt.show()
        print("[ DONE ]")
