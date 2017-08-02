# This script compares two csv files; takes them in as command-line args. It doesn't care if the line orders or
# column orders are different, as corresponding data elements are compared.

import sys

class CSVFile(object):
    def  __init__(self, filename):
        self.filename = filename
        with open(filename, mode='r') as file:
            self.header = file.readline().strip().split(',')
            self.data_lines = [ line.strip().split(',') for line in file.readlines()]

    def to_dict(self, primary_key):
        hsh = {}
        for line_index in range(len(self.data_lines)):
            line = self.data_lines[line_index]
            line_hash = {}
            for header_index in range(len(self.header)):
                line_hash[self.header[header_index]] = line[header_index]
                hsh[line[header_index]] = line_hash
        return hsh

    def same_header(self, other_csv):
        return sorted(self.header) == sorted(other_csv.header)

    def same_entry_count(self, other_csv):
        return len(self.data_lines) == len(other_csv.data_lines)

    def same_data(self, other_csv, primary_key='ID'):
        is_same = True
        this_hsh = self.to_dict(primary_key=primary_key)
        other_hsh = other_csv.to_dict(primary_key=primary_key)
        for unique_key, this_values in this_hsh.iteritems():
            try:
                other_values = other_hsh[unique_key]
            except KeyError as e:
                print('Entry with unique identifier: %s:%s is not in the second file.'  % (primary_key, unique_key))
                raise e
            for item, this_value in this_values.iteritems():
                other_value = other_values[item]
                if this_value != other_value:
                    #print('%s : %s -- item: %s, %s != %s', (primary_key, unique_key, item, this_value, other_value))
                    is_same = False
        return is_same

if __name__ == '__main__':
    file1, file2 = sys.argv[1:]
    file1 = CSVFile(file1)
    file2 = CSVFile(file2)

    print('Same header? %s' % file1.same_header(file2))
    print('Same entry count? %s' % file1.same_entry_count(file2))
    print('Same data? %s ' % file1.same_data(file2))
