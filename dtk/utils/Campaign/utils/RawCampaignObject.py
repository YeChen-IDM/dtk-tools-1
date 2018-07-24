import json
from dtk.utils.Campaign.ClassValidator import ClassValidator


class RawCampaignObject(object):
    _definition = {"json_object": {}, "class": "RawCampaignObject"}
    _validator = ClassValidator(_definition, 'RawCampaignObject')

    def __init__(self, json_object={}):
        self.json_object = self.load_json_object(json_object)

    @staticmethod
    def load_json_object(json_object):
        if isinstance(json_object, str):
            return json.loads(json_object)
        else:
            return json_object

    def get_json_object(self):
        return self.json_object

    def to_json(self, use_defaults=True, human_readability=True):
        from dtk.utils.Campaign.utils.CampaignEncoder import CampaignEncoder
        ec = CampaignEncoder(use_defaults)
        t = ec.encode(self)
        w = json.loads(t)
        if human_readability:
            return json.dumps(w, sort_keys=True, indent=3)
        else:
            return json.dumps(w, sort_keys=True)