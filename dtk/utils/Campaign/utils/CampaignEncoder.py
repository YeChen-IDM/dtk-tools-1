import json
import sys
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
          - Enum
          - bool
          - Campaign class
          - RawCampaignObject
        """

        # handle enum case
        if isinstance(o, Enum):
            return o.name

        # handle Number case
        if isinstance(o, np.int32):
            return int(o)

        if isinstance(o, np.int64):
            return int(o)

        if isinstance(o, RawCampaignObject):
            return o.get_json_object()

        # First get the dict
        d = o.__dict__

        if not hasattr(o, "_definition"):
            raise Exception()

        definition = o._definition

        r = {}
        for key, val in d.items():
            if key not in definition:
                r[key] = val
                continue

            valid = definition[key]

            if isinstance(val, bool):
                if self.Use_Defaults:
                    # For Root Campaign class, output all no matter of Use_Defaults
                    if o.__class__.__name__ == 'Campaign':
                        r[key] = self.convert_bool(val)
                    else:
                        if self.convert_bool(val) != valid.get('default', None):
                            r[key] = self.convert_bool(val)
                else:
                    r[key] = self.convert_bool(val)
            elif isinstance(valid, dict):
                if self.Use_Defaults:
                    # For Root Campaign class, output all no matter of Use_Defaults
                    if o.__class__.__name__ == 'Campaign':
                        r[key] = val
                    else:
                        if isinstance(val, Enum):
                            if val.name != valid.get('default', None):
                                r[key] = val
                        elif val and val != valid.get('default', None):
                            r[key] = val
                else:
                    r[key] = val
            else:
                if self.Use_Defaults:
                    # For Root Campaign class, output all no matter of Use_Defaults
                    if o.__class__.__name__ == 'Campaign':
                        r[key] = val
                    else:
                        if val and val != valid:
                            r[key] = val
                else:
                    r[key] = val

        d = r

        # for Campaign class we defined, don't output class
        if o.__class__.__name__ != 'Campaign':
            d["class"] = o.__class__.__name__

        return d

    @staticmethod
    def convert_bool(val):
        """
        Map: True/False to 1/0
        """
        return 1 if val else 0


