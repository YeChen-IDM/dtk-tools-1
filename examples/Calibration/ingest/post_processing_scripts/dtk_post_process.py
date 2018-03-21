#!/usr/bin/python

from __future__ import print_function

import os # for path
import json
import pandas as pd
import time
import math # ceil

# Relative to DTK output folder:
ByAgeAndGender_filename     = "ReportHIVByAgeAndGender.csv"
Calibration_filename        = "Calibration.json"

def timing(f, message):
    if message is not None:
        print('\r' + message, end='')
    t_start = time.clock()
    result = f()
    t_end = time.clock()
    if message is not None:
        print('{0:>10f}'.format(t_end-t_start))

    return result

def application(output_dir):
    print( "Hello from Python!" );
    print( "Started Python post processing  @ " + time.asctime() )
    print( "Current working directory is: " + os.getcwd() )

    filename = os.path.join( output_dir, ByAgeAndGender_filename )

    if os.path.isfile( filename ) == False :
        print( "!!!! Can't open " + filename +"!" )
        return

    data = timing(lambda: pd.read_csv(filename), message='Load data:    ')

    reports = get_reports(data)

    data.columns = map(lambda s: s.strip().replace(' ', '_').replace('(', '').replace(')', ''), data.columns)
    node_ids = [int(node_id) for node_id in data.NodeId.unique()]
    node_ids.sort()

    segmented = {}
    for year in data.Year.unique():
        segmented[year] = {0: [], 1: []}
        segmented[year][0] = data[(data.Year == year) & (data.Gender == 0)]
        segmented[year][1] = data[(data.Year == year) & (data.Gender == 1)]

    '''
    writer = pd.ExcelWriter(os.path.join(output_dir,'PostProcess.xlsx'), engine='xlsxwriter')
    for report in reports:
        result = timing(lambda: process_report(report, segmented, node_ids), message=report['Name'])  # 'Report: ')
        result.to_excel(writer, sheet_name=report['Name'], merge_cells=False)
    writer.save()
    '''

    post_process_dir = 'post_process'
    if not os.path.exists(os.path.join(output_dir, post_process_dir)):
        os.makedirs(os.path.join(output_dir, post_process_dir))

    for report in reports:
        result = timing(lambda: process_report(report, segmented, node_ids), message=report['Name'])
        result.to_csv(os.path.join(output_dir, post_process_dir, '%s.csv'%report['Name']))

    print( "Finished Python post processing @ " + time.asctime() )

    pass


def get_reports(data):
    entries = []

    first_year = int( math.ceil(data.Year.min()) )
    last_prevalence_year = int(data.Year.max())
    # last_incidence_year = last_prevalence_year - 1

    entry = {'Name': 'Currently_Marital',
             'Type': 'Prevalence',
             'Year': [2015.5],
             'AgeBins': [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Currently_MARITAL) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    entry = {'Name': 'Ever_Marital',
             'Type': 'Prevalence',
             'Year': [2015.5],
             'AgeBins': [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Ever_MARITAL) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    entry = {'Name': 'Current_Partners',
             'Type': 'Prevalence',
             'Year': [2015.5],
             'AgeBins': [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Current_Partners) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    entry = {'Name': 'Lifetime_Partners',
             'Type': 'Prevalence',
             'Year': [2015.5],
             'AgeBins': [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Lifetime_Partners) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    entry = {'Name': 'ProvincialPrevalence',
             'Type': 'Prevalence',
             'Year': range(first_year, last_prevalence_year + 1) + [2013.5],
             'AgeBins': [(15, 25), (15, 50)],
             'Gender': [0, 1],
             'ByNode': True,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Infected) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    entry = {'Name': 'NationalPrevalence',
             'Type': 'Prevalence',
             'Year': [2013.5],
             'AgeBins': [ (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55) ],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Infected) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    entry = {'Name': 'PopScaling',
             'Type': 'Prevalence',
             'Year': [2013.5],
             'AgeBins': [(15, 65)],
             'Gender': [0, 1],
             'ByNode': True,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.Population}
    entries.append(entry)

    entry = {'Name': 'Population',
             'Type': 'Prevalence',
             'Year': range(2002, last_prevalence_year + 1) + [2013.5],
             'AgeBins': [(0, 1), (1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 100)],
             'Gender': [0, 1],
             'ByNode': True,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.Population}
    entries.append(entry)

    entry = {'Name': 'Infected',
             'Type': 'Prevalence',
             'Year': range(2002, last_prevalence_year + 1),
             'AgeBins': [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55,100)],
             'Gender': [0, 1],
             'ByNode': True,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: (row.Infected)}
    entries.append(entry)

    entry = {'Name': 'ANC_Infected',
             'Type': 'Prevalence',
             'Year': range(first_year, last_prevalence_year + 1),
             'AgeBins': [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 100)],
             'Gender': [1],
             'ByNode': True,
             # Percent from Table 4 of 2012 Zimbabwe ANC report
             'Bin_Weights': [0.01, 0.193, 0.303, 0.239, 0.156, 0.077, 0.015, 0.001, 0.0],
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: (row.Infected),
             'Lambda_PostWeighting': lambda row: row[0]}
    entries.append(entry)

    entry = {'Name': 'ANC_Population',
             'Type': 'Prevalence',
             'Year': range(first_year, last_prevalence_year + 1),
             'AgeBins': [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 100)],
             'Gender': [1],
             'ByNode': True,
             # Percent from Table 4 of 2012 Zimbabwe ANC report
             'Bin_Weights': [0.01, 0.193, 0.303, 0.239, 0.156, 0.077, 0.015, 0.001, 0.0],
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.Population,
             'Lambda_PostWeighting': lambda row: row[0]}
    entries.append(entry)

    entry = {'Name': 'Number_On_ART',
             'Type': 'Prevalence',
             'Year': [x + 0.5 for x in xrange(2000,   last_prevalence_year)],
             'AgeBins': [(0, 15), (15, 100)],
             'Gender': [0, 1],
             'ByNode': True,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.On_ART}
    entries.append(entry)

    entry = {'Name': 'Number_Infected',
             'Type': 'Prevalence',
             'Year': [x + 0.5 for x in xrange(2000,   last_prevalence_year)],
             'AgeBins': [(0, 15), (15, 100)],
             'Gender': [0, 1],
             'ByNode': True,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.Infected}
    entries.append(entry)

    entry = {'Name': 'Number_On_ART_for_Suppression',
             'Type': 'Prevalence',
             'Year': [x + 0.5 for x in xrange(2000,   last_prevalence_year)],
             'AgeBins': [(0, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55,100)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.On_ART}
    entries.append(entry)

    entry = {'Name': 'Number_Infected_for_Suppression',
             'Type': 'Prevalence',
             'Year': [x + 0.5 for x in xrange(2000,   last_prevalence_year)],
             'AgeBins': [(0, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55,100)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: row.Infected}
    entries.append(entry)

    entry = {'Name': 'Tested_Ever',
             'Type': 'Prevalence',
             'Year': range(first_year, last_prevalence_year + 1),
             'AgeBins': [(15, 20), (20, 25), (25, 30), (30, 40), (40, 50)],
             'Gender': [0, 1],
             'ByNode': False,
             'Map': lambda rows: rows.sum(),
             'Reduce': lambda row: float(row.Tested_Ever) / float(row.Population) if row.Population > 0 else 0}
    entries.append(entry)

    return entries

def process_report(report, segmented, node_ids):

    output = pd.DataFrame(columns=['Year', 'Node', 'Gender', 'AgeBin', 'Result'])

    for year in report['Year']:
        for gender in report['Gender']:
            gender_str = 'Male' if gender == 0 else 'Female'
            data = segmented[year][gender]
            for min_age, max_age in report['AgeBins']:
                if report['ByNode']:
                    for node_id in node_ids:
                        rows = timing(lambda: data[(data.NodeId == node_id) &
                                                   (data.Age >= min_age) &
                                                   (data.Age < max_age)], message=None)  # 'Select rows: ')
                        mapping = timing(lambda: report['Map'](rows), message=None)  # 'Sum data: ')
                        try:
                            result = report['Reduce'](mapping)
                        except AttributeError as e:
                            print(' -- FAILED!')
                            return output
                        output.loc[output.shape[0],:] =  (year, node_id, gender_str, '[%d, %d)'%(min_age, max_age), result)
                else:
                    rows = timing(lambda: data[(data.Age >= min_age) & (data.Age < max_age)], message=None)  # 'Select rows: ')
                    mapping = timing(lambda: report['Map'](rows), message=None)  # 'Sum data: ')
                    try:
                        result = report['Reduce'](mapping)
                    except AttributeError as e:
                        print(' -- FAILED!')
                        return output
                    output.loc[output.shape[0],:] =  (year, 'All', gender_str, '[%d, %d)'%(min_age, max_age), result)

    output.set_index(['Year', 'Node', 'Gender', 'AgeBin'], inplace=True)

    return output


if __name__ == '__main__':
    application( 'output')
    pass
