import json


class ClassDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.json_to_class)
        self.class_dict = {}
        self.class_params = {}
        self.enum_dict = {}

        self.test_dict = {}
        self.list_dict = {}
        self.vcc = []

    def json_to_class(self, d):

        # args will be used to dynamically create class later
        self.parse_for_args(d)

        # schema storing classes only
        self.parse_for_class(d)

        # retrieve all enums
        self.parse_for_enum(d)

        # do nothing (transformation)
        return d

    def parse_for_enum(self, d):

        for key, value in d.items():
            if key.startswith('idmType:'):
                continue

            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, dict) and 'enum' in v:
                        enum_key = "{}_{}".format(key, k)
                        self.enum_dict[enum_key] = v['enum']

    def parse_for_class(self, d):
        if 'class' not in d:
            return

        class_name = d.get('class')

        # in case we want to do some transformation
        args = {
            key: value
            for key, value in d.items()
        }

        self.class_dict[class_name] = args

    def parse_for_args(self, d):
        if 'class' not in d:
            return

        args = []
        defaults = []
        for key, value in d.items():
            if key == 'class':
                continue    # skip and move to next

            # store keys
            args.append(key)

            # store default values
            if isinstance(value, list):
                defaults.append(value)
            elif isinstance(value, dict):
                if 'default' in value:
                    v = value['default']
                    if isinstance(v, str):
                        defaults.append('{}'.format(v))
                    else:
                        defaults.append(v)
                elif key == 'Waning_Config':        # specially handle case!
                    defaults.append([])
                else:
                    defaults.append(None)
            elif isinstance(value, str):
                defaults.append('{}'.format(value))
            else:
                defaults.append(value)

        class_name = d.get('class')
        self.class_params[class_name] = {"args": args, "defaults": defaults, "_definition": d}

    # override method
    def decode(self, json_str):

        # so that we may call decode many times!
        self.class_dict = {}
        self.class_params = {}
        self.enum_dict = {}

        return json.JSONDecoder.decode(self, json_str)
