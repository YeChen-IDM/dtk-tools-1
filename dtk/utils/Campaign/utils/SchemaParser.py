import json
from dtk.utils.Campaign.utils.SchemaDecoder import SchemaDecoder


class SchemaParser:
    """
    Hold a SchemaDecoder to process Schema transformation
    """
    def __init__(self, schema_file):
        self.schema_decoder = None
        self.schema_file = schema_file
        self.load_schema(schema_file)

    def load_schema(self, schema_file):
        schema_data = self.load_json_data(schema_file)
        idmType_dict = {k: v for k, v in schema_data['idmTypes'].items()}
        self.schema_decoder = SchemaDecoder(idmType_dict)

    @staticmethod
    def load_json_data(schema_file):
        # http://www.neuraldump.net/2017/06/how-to-suppress-python-unittest-warnings/
        import warnings
        warnings.simplefilter("ignore", ResourceWarning)
        try:
            return json.load(open(schema_file, 'rb'))
        except IOError:
            raise Exception('Unable to read file: %s' % schema_file)

    @staticmethod
    def save_schema_to_file(filename, content):
        f = open('{}.json'.format(filename), 'w')
        f.write(content)
        f.close()

    def parse(self, schema_path=None):
        if schema_path is None:
            schema_path = self.schema_file

        schema_data = self.load_json_data(schema_path)
        json_str = json.dumps(schema_data, indent=3)

        schema = self.schema_decoder.decode(json_str)
        return schema