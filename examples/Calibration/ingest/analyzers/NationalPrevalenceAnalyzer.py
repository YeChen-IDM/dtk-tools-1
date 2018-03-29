import logging
import pandas as pd
import os, sys
import numpy as np
from scipy.stats import beta
from scipy.special import gammaln

from dtk.utils.observations.DataFrameWrapper import DataFrameWrapper
from dtk.utils.observations.PopulationObs import PopulationObs
from analyzers.PostProcessAnalyzer import PostProcessAnalyzer

# Plotting
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns

logger = logging.getLogger(__name__)

class NationalPrevalenceAnalyzer(PostProcessAnalyzer):

    reference_key = 'NationalPrevalence'
    sim_reference_key = 'Prevalence'
    log_float_tiny = np.log( np.finfo(float).tiny )

    def __init__(self, site, weight, **kwargs):
        super(NationalPrevalenceAnalyzer, self).__init__(**kwargs)

        self.filenames += [ os.path.join('output', 'post_process', 'NationalPrevalence.csv') ]

        self.name = self.__class__.__name__
        self.weight = weight
        self.setup = {}

        self.site = site
        self.reference = self.site.reference_data
        self.alpha_channel, self.beta_channel = self.reference.add_beta_parameters(channel=self.reference_key)
        self.reference = self.reference.filter(keep_only=[self.reference_key, self.alpha_channel, self.beta_channel])

    def filter(self, sim_metadata):
        ret = super(NationalPrevalenceAnalyzer, self).filter(sim_metadata)
        return ret


    def apply(self, parser):
        ret = super(NationalPrevalenceAnalyzer, self).apply(parser)

        pop_scaling = ret['Pop_Scaling']
        sim = parser.raw_data[self.filenames[-1]] \
            .drop('Node', axis=1) \
            .rename(columns={'Result':self.sim_reference_key})

        stratifiers = ['Year', 'Gender', 'AgeBin']
        sim_dfw = DataFrameWrapper(dataframe=sim, stratifiers=stratifiers)
        merged = self.reference.merge(sim_dfw, index=stratifiers,
                                      keep_only=[self.reference_key, self.sim_reference_key, self.alpha_channel, self.beta_channel])

        merged_years = merged.get_years()
        reference_years = self.reference.get_years()

        if reference_years != merged_years:
            raise Exception("[%s] Failed to find all data years (%s) in simulation output (%s)." % (self.name, reference_years, merged_years))

        # If analyzing simulation not generated by itertool, __sample_index__ will not be in tags
        # Instead, use Run_Number
        sample = parser.sim_data.get('__sample_index__')
        if sample is None:
            sample = parser.sim_data.get('Run_Number')

        merged = merged._dataframe
        merged.index.name = 'Index'

        shelve_data = {
            'Data': merged,
            'Sim_Id': parser.sim_id,
            'Sample': sample
        }
        self.shelve_apply( parser.sim_id, shelve_data)

        if self.debug:
            print("size (MB):", sys.getsizeof(shelve_data)/8.0/1024.0)

    def compare_year_gender(self, sample):
        # Note: Might be called extra times by pandas on apply for purposes of "optimization"
        # http://stackoverflow.com/questions/21635915/why-does-pandas-apply-calculate-twice

        a = sample[self.alpha_channel]
        b = sample[self.beta_channel]
        x = sample[self.sim_reference_key]

        # v1 = gammaln(a)
        # v2 = gammaln(b)
        # t = a+b
        # v3 = gammaln(t)
        # This is what we're calculating:
        # BETA(output_i | alpha=alpha(Data), beta = beta(Data) )
        betaln = np.multiply((a-1), np.log(x)) \
            + np.multiply((b-1), np.log(1-x)) \
            - (gammaln(a)+gammaln(b)-gammaln(a+b))

        # Replace -inf with log(machine tiny)
        betaln[ np.isinf(betaln) ] = self.log_float_tiny

        # Scaling

        x_mode = np.divide((a - 1), (a + b - 2))
        largest_possible_log_of_beta = beta.pdf(x_mode, a, b)
        scale_max = 15
        beta_ratio = np.divide (scale_max, largest_possible_log_of_beta)

        betaln = np.multiply (betaln, beta_ratio)

        # betaln = max(betaln, self.log_float_tiny)

        return betaln

    def compare(self, sample):
        LL = sample.reset_index().groupby(['Year', 'Gender']).apply(self.compare_year_gender)
        return (sum(LL.values)*self.weight)

    def combine(self, parsers):
        '''
        Combine the simulation data into a single table for all analyzed simulations.
        '''
        shelved_data = super(NationalPrevalenceAnalyzer, self).combine(parsers)

        if shelved_data is not None:
            if self.verbose:
                print('Combine from cache')
            self.data = shelved_data['Data']
            return

        selected = [ self.shelve[str(sim_id)]['Data'] for sim_id in self.sim_ids ]
        keys = [ (self.shelve[str(sim_id)]['Sample'], self.shelve[str(sim_id)]['Sim_Id'])
            for sim_id in self.sim_ids ]

        self.data = pd.concat( selected, axis=0,
                            keys=keys,
                            names=['Sample', 'Sim_Id'] )

        self.data.reset_index(level='Index', drop=True, inplace=True)

        try:
            self.shelve_combine({'Data':self.data})
        except:
            print("shelve_combine didn't work, sorry")


    def cache(self):
        # print('CACHE')
        # '''
        # Return a cache of the minimal data required for plotting sample comparisons
        # to reference comparisons.
        # '''
        # d = self.data.reset_index()
        # return d.where((pd.notnull(d)), None).to_dict(orient='list')
        pass

    def uid(self):
        print('UID')
        ''' A unique identifier of site-name and analyzer-name. '''
        return '_'.join([self.site.name, self.name])


    ###########################################################################
    def plotBestAndWorst(self, **kwargs):
        data = kwargs.pop('data')

        best_data = data.set_index('Sample').loc[self.best_sample].reset_index()
        #sns.pointplot( data=best_data, x=kwargs['x'], y=kwargs['y'], hue=kwargs['hue'], palette=sns.color_palette("Greens"), markers='o', linestyles='-', join=True, scale=kwargs['scale'])
        sns.pointplot( data=best_data, x=kwargs['x'], y=kwargs['y'], color='g', markers='o', linestyles='-', join=True, scale=kwargs['scale'])

        worst_data = data.set_index('Sample').loc[self.worst_sample].reset_index()
        #sns.pointplot( data=worst_data, x=kwargs['x'], y=kwargs['y'], hue=kwargs['hue'], palette=sns.color_palette("Reds"), markers='o', linestyles='-', join=True, scale=kwargs['scale'])
        sns.pointplot( data=worst_data, x=kwargs['x'], y=kwargs['y'], color='r', markers='o', linestyles='-', join=True, scale=kwargs['scale'])

    def make_collection(self, d):
        return zip(d['Year'], d['Sim_Prevalence'])


    def plot_agebin(self, data):
        agebin = data['AgeBin'].values[0] # Could be more efficient
        data.set_index('Gender', inplace=True)
        genders = data.index.unique().values.tolist()

        ref = self.reference.reset_index().set_index(['AgeBin', 'Gender']).loc[agebin]

        fig, ax = plt.subplots(1, len(genders), figsize=(16,10), sharey='row', sharex='row')
        for gender, a in zip(genders, ax):
            data_g = data.loc[[gender]]
            ref_g = ref.loc[gender]

            # Color by result?
            data_g_by_sim_id = data_g.groupby('Sim_Id')
            lc = mc.LineCollection( data_g_by_sim_id.apply(self.make_collection), linewidths=0.1, cmap=plt.cm.jet )
            #lc.set_array( data_g_by_sim_id['Results'].apply(lambda z:z)) # <-- Hopefully same order?

            a.add_collection(lc)

            # Use Count to make a poisson confidence interval (noraml approx)
            a.plot(ref_g['Year'], ref_g['NationalPrevalence'], 'k.', ms=25)

            a.autoscale()
            a.margins(0.1)
            a.set_title('%s: %s'%(gender,agebin))
        return fig, agebin


    def plot(self):

        return

        # TODO: Make some nice plots ...

        if '_iter' in self.exp_name:
            toks = self.exp_name.split('_iter')
            self.basedir = toks[0]
            iterdir = 'iter%d'%int(float(toks[1]))
            figdir = os.path.join(self.basedir, iterdir, self.name) # TODO! # calib_manager.iteration_directory()
        else:
            figdir = os.path.join(self.working_dir, self.basedir, self.exp_id, self.name)
        if not os.path.isdir(figdir):
            os.mkdir(figdir)

        merged = self.merged.reset_index()

        ''' PUT ME BACK???
        ranked_samples = merged.groupby(['Sample']).mean().sort_values(['Results'], ascending=False)
        self.ranked_samples = ranked_samples.reset_index()['Sample'].values

        sorted_results = merged.groupby('Sample').mean()['Results']
        self.best_sample = np.argmax(sorted_results)
        self.worst_sample = np.argmin(sorted_results)
        '''

        merged['Sim_Prevalence'] = 100*merged['Infected']/merged['Population']

        print(merged.head())
        print(merged['Year'].unique().tolist())
        print(merged.dtypes)
        for fig, agebin in merged.groupby('AgeBin').apply(self.plot_agebin):
            fig.savefig(os.path.join(figdir, 'National Prevalence %s.%s'%(agebin, self.fig_format)));
            plt.close(fig)


        # ----------------------------------------------------------------------

        ''' TOO SLOW:

        # TODO:
        # * Error bars on barplot of data
        # * Color order by log likelihood
        # * legend
        g = sns.factorplot(x='AgeBin', y='NationalPrevalence', data=merged, col='Gender', row='Year', kind='bar', color='#A9A9A9', size=4, aspect=1.5, margin_titles=True)
        #h_bar, l_bar = g.axes.flat[0].get_legend_handles_labels() # Empty because no hue

        nSamples = len(set(merged['Sample'].values))
        g.map_dataframe(self.plotBestAndWorst, x='AgeBin', y='Sim_Prevalence', hue='Sample', zorder=100, scale=1 ) # inner='point'
        g.map_dataframe(sns.pointplot, x='AgeBin', y='Sim_Prevalence', hue='Sample', palette=sns.color_palette("coolwarm", nSamples), zorder=-100, join=True, scale=0.1 ) # inner='point'
        g = g.set_titles("{col_name}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('National Prevalence', fontsize=16)

        # Axis outside right
        #handles, labels = g.axes.flat[0].get_legend_handles_labels()    # Draw nice legend outside to the right
        #plt.legend([h_bar[0], h_points[0]], [l_bar[0], l_points[0]], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        g.savefig(os.path.join(figdir, 'National Prevalence.' + self.fig_format)); plt.close()

        plt.figure()
        sns.distplot(ranked_samples['Results'])
        plt.savefig(os.path.join(figdir, 'National Prevalence Histogram.' + self.fig_format)); plt.close()

        plt.figure()
        rs = ranked_samples.reset_index()
        plt.scatter(rs['Results'], rs['Sample'])
        plt.plot(ranked_samples.iloc[0]['Results'], ranked_samples.index[0], 'go')
        plt.plot(ranked_samples.iloc[-1]['Results'], ranked_samples.index[-1], 'ro')
        plt.ylim([-0.5, nSamples-0.5])
        print('F'*80)
        plt.savefig(os.path.join(figdir, 'National Prevalence LL.' + self.fig_format)); plt.close()
        '''

        print("NationalPrevalencePlotter::DONE")
    ###########################################################################


    def finalize(self):
        fn = 'Results_%s.csv'%self.__class__.__name__
        out_dir = os.path.join(self.working_dir, self.basedir, self.exp_id)
        print('--> Writing %s to %s'%(fn, out_dir))
        NationalPrevalenceAnalyzer.mkdir_p(out_dir)
        results_filename = os.path.join(out_dir, fn)
        self.data.to_csv(results_filename)

        # Call 'compare' ... TODO: Check!
        self.result = self.data.reset_index().groupby(['Sample']).apply(self.compare)

        # Close the shelve file, among other little things.  Can take a long time:
        super(NationalPrevalenceAnalyzer, self).finalize()
