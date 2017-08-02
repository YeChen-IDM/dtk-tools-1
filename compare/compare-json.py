"""
This script is for comparing two .json files for identical content. key order is ignored. The first .json file
is the 'standard' to which the second file (test file) is compared against. Extraneous keys in file2 as well as
keys missing from file2 (relative to file1, the standard) are reported, as well as any value differences found
in non-dict entries in the json/dict structure.
"""

import json
import sys

def keys_in_first_and_not_in_second(keys1, keys2):
    missing_keys = []
    in_common_keys = []
    for key in keys1:
        if key not in keys2:
            missing_keys.append(key)
        else:
            in_common_keys.append(key)
    return missing_keys, in_common_keys

def recursive_compare(dict1, dict2, path):
    are_same = True # default
    str_path = '.'.join(path)
    if type(dict1) != type(dict2):
        print('path: %s are not the same types (%s, %s)' % (str_path, type(dict1), type(dict2)))
        are_same = False
    elif not isinstance(dict1, dict):
        are_same = (dict1 == dict2)
        if not are_same:
            print('path: %s are type %s and are not the same:\n\%s\nVS.\n%s' % (str_path, type(dict1), dict1, dict2))
    else: # they are both dicts, recurse!
        keys1 = dict1.keys()
        keys2 = dict2.keys()
        missing_keys, in_common_keys = keys_in_first_and_not_in_second(keys1, keys2)
        if len(missing_keys) > 0:
            print('path: %s are type dict but these keys are missing from the test json: %s' %
                  (str_path, missing_keys))
            are_same = False
        extraneous_keys, _ = keys_in_first_and_not_in_second(keys2, keys1)
        if len(missing_keys) > 0:
            print('path: %s are type dict but these unnecessary keys are in the test json: %s' %
                  (str_path, extraneous_keys))
            are_same = False

        # recurse on the in-common-keys to report on as much as possible in between the two dicts
        for next_key in in_common_keys:
            new_path = path+[next_key]
            print('Recursing into: %s' % new_path)
            are_same = recursive_compare(dict1[next_key], dict2[next_key], path=new_path) and are_same
    print('are_same? %s' % are_same)
    return are_same

def compare(dict1, dict2):
    """
    Assumes dict1 is the canonical 'correct' copy
    :param dict1: a dict from a json file
    :param dict2: a dict from another json file we want to compare against the 'correct' dict1
    :return: True/False, are the dicts the same?
    """
    are_same = True
    for item in dict1:
        if item in dict2:
            are_same = are_same and recursive_compare(dict1[item], dict2[item], path=[item])
        else:
            print('item: %s in standard json file is not in comparison json file at the root level.' % item)
    return are_same

if __name__ == '__main__':
    file1, file2 = sys.argv[1:]

    with open(file1, 'r') as f:
        dict1 = json.loads(f.read())
    with open(file2, 'r') as f:
        dict2 = json.loads(f.read())

    are_same = compare(dict1, dict2)
    print('***\nFiles are the same? %s' % are_same)
