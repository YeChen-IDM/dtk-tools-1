import json
import sys
from dtk.utils.Campaign.CampaignClass import *
from dtk.utils.Campaign.utils.RawCampaignObject import RawCampaignObject


class CampaignDecoder(json.JSONDecoder):
    """
    JSON to Class
    """
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.json_to_campaign)

    def json_to_campaign(self, d):

        inst = None

        if 'class' in d:
            # get class name
            class_name = d['class']

            # find class definition
            _class = globals().get(class_name, None)

            # remove 'class' if it is normal class (defined in Schema)
            if _class:
                d.pop('class')

            args = {}
            for key, value in d.items():
                k = "{}_{}_Enum".format(class_name, key)
                is_bool = self.is_bool_type(key, value, _class)

                if k in globals():
                    _enum = globals()["{}_{}_Enum".format(class_name, key)]
                    args[key] = _enum[value]
                elif is_bool:
                    args[key] = True if value else False
                else:
                    args[key] = value

            try:
                if _class:
                    inst = _class(**args)
                else:
                    # add class back
                    d['class'] = class_name
                    inst = RawCampaignObject(d)
            except Exception as ex:
                raise ex
        else:
            inst = d

        return inst

    @staticmethod
    def is_bool_type(key, value, _class):

        try:
            if _class is None:
                return False

            if key not in _class._definition:
                return False

            valid = _class._definition[key]
        except Exception as ex:
            # print(ex)
            raise ex

        if not isinstance(valid, dict):
            return False

        value_type = valid.get('type', None)

        if value_type is None:
            return False        # no validation

        if value_type in ('bool', 'Bool', 'boolean'):
            return True
        else:
            return False
