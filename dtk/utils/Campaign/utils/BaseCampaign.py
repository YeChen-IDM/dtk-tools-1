import json
from enum import Enum
from dtk.utils.Campaign.utils.CampaignEncoder import CampaignEncoder
from dtk.utils.Campaign.utils.RawCampaignObject import RawCampaignObject


# Turn On/Off class members validation
VALIDATE = True


class BaseCampaign:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        if VALIDATE:
            self._validator.validate(key, value)

        # Special case
        if isinstance(value, RawCampaignObject):
            super().__setattr__(key, value)
            return

        valid = self._definition.get(key)
        if not isinstance(valid, dict):
            super().__setattr__(key, value)
            return

        value_type = valid.get('type', None)

        if value_type is None:
            super().__setattr__(key, value)
            return

        if value_type in ('bool', 'Bool', 'boolean'):
            if isinstance(value, bool):
                super().__setattr__(key, value)
            elif isinstance(value, int) or isinstance(value, float):
                super().__setattr__(key, True if value else False)
        elif value_type in ('enum', 'Enum'):
            if isinstance(value, str):
                super().__setattr__(key, value)
            elif isinstance(value, Enum):
                super().__setattr__(key, value.name)
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def to_json(self, use_defaults=True, human_readability=True):
        ec = CampaignEncoder(use_defaults)
        t = ec.encode(self)
        w = json.loads(t)
        if human_readability:
            return json.dumps(w, sort_keys=True, indent=3)
        else:
            return json.dumps(w, sort_keys=True)

    def save_to_file(self, filename=None, use_defaults=True):
        if filename is None:
            filename = 'output_class'
        content = self.to_json(use_defaults=use_defaults)
        f = open('{}.json'.format(filename), 'w')
        f.write(content)
        f.close()