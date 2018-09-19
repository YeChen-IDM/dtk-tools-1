import os
from dtk.utils.Campaign.utils.CampaignManager import CampaignManager

"""
This script allows to run exactly what the `dtk generate_classes` would run but in a convenient .py file.
Simply adjust the following variables to your environment:
- exe_path: where to find the Eradication executable for which we want to generate files
- output_path: where to create the generated files 
- debug: Do we want to keep the intermediate files?
"""

debug = False
exe_path = r"F:\Temp\builds\1202\Eradication.exe"
output_path = r"F:\Temp\builds\1202"

def validate_inputs(exe_path, output_path):
    """
    Validate inputs
    :param exe_path:
    :param output_path:
    :return:
    """
    file_exits = os.path.isfile(exe_path)
    folder_exists = os.path.isdir(output_path)

    # check file exists
    if not file_exits:
        print("File '{}' doesn't exist!".format(exe_path))
        return False

    # check output location exists
    if not folder_exists:
        print("Output Location '{}' doesn't exist!".format(output_path))
        return False

    # display inputs information
    print("\nEXE Path: ", exe_path)
    print("Output Location: ", output_path)

    return True


if __name__ == "__main__":
    valid = validate_inputs(exe_path, output_path)
    if valid:
        CampaignManager.generate_campaign_classes(exe_path, os.path.abspath(output_path), debug)
        print('\nCampaign Classes successfully generated from Eradication EXE!')
