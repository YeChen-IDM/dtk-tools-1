import json


class SchemaDecoder(json.JSONDecoder):
    """
    Transform Eradication schema to a 'standard' format
    """
    def __init__(self, idmType_dict={}):
        json.JSONDecoder.__init__(self, object_hook=self.json_to_campaign)
        self.idmType_dict = idmType_dict

    def json_to_campaign(self, d):
        g = {}

        for key, value in d.items():
            if isinstance(value, list):
                self.handle_list(key, value, g)
            elif isinstance(value, dict):
                self.handle_dict(key, value, g)
            else:
                g[key] = value

        return g

    def handle_list(self, key, value, g):

        def all_dict(key, value):
            for i in value:
                if not isinstance(i, dict):
                    return False

            return True

        if key in ("Sim_Types", "enum", 'default', 'Built-in'):
            g[key] = value
            return

        # special case
        if len(value) == 1 and isinstance(value[0], dict):
            self.handle_list_dict_case(key, value, g)
            return

        if all_dict(key, value):
            obj_type = self.get_type(value[0]) if len(value) == 1 else 'object'
            g[key] = {
                "type": 'list',
                "item_type": {
                    "type": obj_type
                },
                "default": []
            }
        else:
            g[key] = {
                "type": 'list',
                "item_type": {
                    "type": 'object'
                },
                "default": []
            }

    @staticmethod
    def get_type(obj):
        if isinstance(obj, int):
            return 'int'
        elif isinstance(obj, float):
            return 'float'
        elif isinstance(obj, list):
            return 'list'
        elif isinstance(obj, str):
            return 'string'
        elif isinstance(obj, dict):
            if 'type' in obj:
                return obj['type']
            else:
                return 'object'
        else:
            return 'object'

    def handle_dict(self, key, value, g):

        if 'base' in value:
            # actually, in this cse 'base' is the only key!
            inst = value
        else:
            if self.check_number_type(value):
                inst = self.handle_default_value(key, value)
            elif self.check_idmType(value):
                obj_type = value.get("type", "") if isinstance(value, dict) else ""
                if obj_type == "idmType:WaningEffect":
                    inst = value
                else:
                    inst = self.handle_dict_has_idmType(key, value)
            else:
                inst = self.handle_dict_no_idmType(key, value)

        g[key] = inst

    def handle_list_dict_case(self, key, value, g):

        it = value[0]

        if len(it) == 2 and '<key>' in it and '<value>' in it:
            item_type = {"type": "dict"}
            for k, v in it.items():
                k2 = k.strip("><")
                k2 = "{}_type".format(k2)
                item_type[k2] = v

            g[key] = {

                "type": 'list',
                "item_type": item_type,
                "default": []
            }
        else:
            item_type = it.get("type", None)
            if item_type:
                g[key] = {
                    "type": 'list',
                    "item_type": it,
                    "default": []
                }
            else:
                g[key] = {
                    "type": 'list',
                    "item_type": {
                        'type': 'object'
                    },
                    "default": []
                }

    def copy_idmType_object(self, item):
        result = None
        if self.check_idmType(item):
            idmType = item['type']
            new_key = idmType[8:]

            try:
                if idmType in self.idmType_dict:
                    v = self.idmType_dict[idmType]
                    if v is None or ('class' not in v and 'base' not in v):
                        result = {new_key: v}
            except Exception as ex:
                raise ex
                # print(idmType)

        result = self.apply_decoder(result)  # very important!!
        if result:
            # get the first key (only key) and return its value
            for k, v in result.items():
                result = v
                break

        return result

    def check_number_type(self, value):
        value_type = value.get('type', None)
        return value_type in ('int', 'float')

    def handle_default_value(self, key, value):
        """
        Workaround of Schema bug: default value is not between min and max

        Note: we take min or max as the default value instead
        """
        value_type = value.get('type', None)

        if value_type is None:
            return value        # no validation

        d = value
        if value_type in ('int', 'float'):
            if 'default' not in value:
                return value

            default = value["default"]
            if 'min' in value and default < value["min"]:
                d["default"] = value["max"]

            if 'max' in value and default > value["max"]:
                d["default"] = value["max"]

        return d

    def check_idmType(self, value):
        """
            "Larval_Habitat_Types": {
              "description": "A measure of ......",
              "type": "idmType:LarvalHabitats"
            }
        :param value:
        :return:
        """
        obj_type = value.get("type", "") if isinstance(value, dict) else ""
        if obj_type.startswith("idmType"):
            return True
        else:
            return False

    def handle_dict_has_idmType(self, key, value):

        obj_type = value['type']
        key = obj_type[8:]

        # copy everything except 'type
        args = {k: v for k, v in value.items() if k != 'type'}

        obj = self.copy_idmType_object(value)
        if obj:
            if isinstance(obj, dict):
                args.update(obj)
            else:
                args = obj
        else:
            args["type"] = 'object'
            args["subclasses"] = key

        return args

    def handle_dict_no_idmType(self, key, value):

        valid = True
        for k, v in value.items():
            if not isinstance(v, list):
                valid = False
                break

        # special case
        # if len(value) == 2 and 'Times' in value and 'Values' in value:
        if valid:
            result = {}
            for k, v in value.items():
                result[k] = {
                    'type': 'list',
                    'item_type': v[0],
                    "default": []
                }

                if 'description' in v[0]:
                    result[k]['description'] = v[0]['description']
                    del result[k]['item_type']['description']
        else:
            result = value

        return result

    def apply_decoder(self, schema_data):
        json_str = json.dumps(schema_data, indent=3)
        schema = self.decode(json_str)

        return schema
