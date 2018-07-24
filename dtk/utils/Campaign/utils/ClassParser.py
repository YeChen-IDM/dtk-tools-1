import json
from dtk.utils.Campaign.utils.ClassDecoder import ClassDecoder


class ClassParser:
    """
    Hold a CampaignDecoder to locate all classes and enums will be used
         create CampaignClass.py and CampaignEnum.py
    """
    def __init__(self, schema_file):
        self.class_decoder = ClassDecoder()
        self.schema_file = schema_file

    @staticmethod
    def load_json_data(schema_file):
        try:
            return json.load(open(schema_file, 'rb'))
        except IOError:
            raise Exception('Unable to read file: %s' % schema_file)

    @staticmethod
    def save_schema_to_file(filename, content):
        f = open('{}.json'.format(filename), 'w')
        f.write(content)
        f.close()

    def parse(self, test_schema=None):
        if test_schema is None:
            test_schema = self.schema_file

        schema_data = self.load_json_data(test_schema)
        json_str = json.dumps(schema_data, indent=3)
        schema = self.class_decoder.decode(json_str)

        return schema