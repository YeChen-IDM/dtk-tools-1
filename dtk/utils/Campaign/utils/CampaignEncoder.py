import json
from enum import Enum
import numpy as np
from dtk.utils.Campaign.utils.RawCampaignObject import RawCampaignObject


class CampaignEncoder(json.JSONEncoder):
    """
    Class to JSON
    """

    def __init__(self, use_defaults=True):
        super(CampaignEncoder, self).__init__()
        self.Use_Defaults = use_defaults

    def default(self, o):
        """
        Specially handle cases:
          - np.int32 and np.int64
          - Enum
          - bool
          - Campaign class
          - RawCampaignObject
        """

        # handle enum case
        if isinstance(o, Enum):
            return o.name

        # handle Number case
        if isinstance(o, np.int32) or isinstance(o, np.int64):
            return int(o)

        if isinstance(o, RawCampaignObject):
            return o.get_json_object()

        # First get the dict
        object_dict = o.__dict__

        # If the object does not have a _definition attribute, we cannot continue
        if not hasattr(o, "_definition"):
            raise Exception("Parsing cannot continue as the object provided does not have a _definition")

        # Root campaign ? If we are at the root of the campaign, we will output parameters regardless of defaults
        campaign_root = o.__class__.__name__ == 'Campaign'

        # Retrieve the object definition
        definition = o._definition

        result = {}
        for key, val in object_dict.items():
            # If the attribute is not in the definition, we assume it is a user-defined
            # attribute and simply add it to the return dict
            if key not in definition:
                result[key] = val
                continue

            # Retrieve the default value from the definition for the current field
            validation = definition[key]
            default_value = validation.get('default', None) if isinstance(validation, dict) else validation

            # If the value is a boolean
            if isinstance(val, bool):
                converted_value = self.convert_bool(val)
                if campaign_root or not (self.Use_Defaults and converted_value == default_value):
                    result[key] = converted_value

            elif isinstance(val, Enum):
                if campaign_root or not (self.Use_Defaults and val.name == default_value):
                    result[key] = val

            else:
                if campaign_root or not (self.Use_Defaults and val == default_value):
                    result[key] = val

        # for Campaign class we defined, don't output class
        if not campaign_root:
            result["class"] = o.__class__.__name__

        return result

    @staticmethod
    def convert_bool(val):
        """
        Map: True/False to 1/0
        """
        return 1 if val else 0


