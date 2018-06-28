import json
from enum import Enum
from simtools.Utilities.Encoding import NumpyEncoder


class ClassValidator:
    """
    Validate class members against class definition
    """

    def __init__(self, definition, cls_name=None):
        self.definition = definition
        self.cls_name = cls_name

    def output_definition(self, details=None):

        details = details or self.definition
        if isinstance(details, dict):
            return json.dumps(details, indent=3, cls=NumpyEncoder)
        elif isinstance(details, list):
            details = [" - {}".format(i) for i in details if i != 'class']
            return '\n'.join(details)
        else:
            return details

    def validate(self, key, value):
        # print(self.cls_name, ' :: ', key, ' ~~ ', value)
        if key in self.definition:
            valid = self.definition[key]

            if isinstance(valid, list):
                self.validate_list(key, value)
            elif isinstance(valid, dict):
                self.validate_dict(key, value)
            else:
                self.validate_other(key, value)
        else:
            # For now we just take user's inputs and not throw exception
            pass
            # raise Exception("'{}' is not a member of the given class ({}). Please check available member list: "
            #                 "\n\n{} ".format(key, self.cls_name, self.output_definition(list(self.definition.keys()))))

    def validate_dict(self, key, value):
        """
        value definition is dict
        """
        valid = self.definition[key]
        value_type = valid.get('type', None)

        if value_type is None:
            return         # no validation

        # member definition details
        json_script = {key: valid}

        if value_type in ('bool', 'Bool', 'boolean'):
            if not isinstance(value, bool):
                if isinstance(value, int) or isinstance(value, float):
                    return  # no validation here, but will cast value to bool type in __setattr__ (CampaignBaseClass)

                raise Exception("'{}' value ({}) is not a bool. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))
        elif value_type in ('int', 'float'):
            if value is None:
                return

            if 'max' in valid and value > valid["max"]:
                raise Exception("'{}' value ({}) is TOO HIGH. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))

            if 'min' in valid and value < valid["min"]:
                raise Exception("'{}' value ({}) is TOO LOW. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))
        elif value_type in ('str', 'string'):
            if not isinstance(value, str):
                raise Exception("'{}' value ({}) is not a string. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))
        elif value_type in ('enum', 'Enum'):
            enum_name = "{}_{}_Enum".format(self.cls_name, key)
            if not isinstance(value, Enum):
                if isinstance(value, str):
                    if value not in valid['enum']:
                        raise Exception(
                            "'{}' value ({}) is not a member of enum. Please check the class member details: "
                            "\n\n{} ".format(key, value, self.output_definition(json_script)))
                else:
                    raise Exception("'{}' value ({}) is not an enum. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))
            elif value.__class__.__name__ != enum_name:
                raise Exception("'{}' value ({}) is not same enum type as defined. Please check the class member details: "
                                    "\n\n{} ".format(key, value, self.output_definition(json_script)))
            else:
                pass
        elif value_type in ('dict', 'Dict'):
            if not isinstance(value, dict):
                raise Exception("'{}' value ({}) is not a dict. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))
        elif value_type in ('list', 'Dynamic String Set'):
            if value and not isinstance(value, list):
                raise Exception("'{}' value ({}) is not a list. Please check the class member details: "
                                "\n\n{} ".format(key, value, self.output_definition(json_script)))
        else:
            pass

        return None

    def validate_list(self, key, value):
        """
        value definition is list
        """
        valid = self.definition[key]

        if value and not isinstance(value, list):
            json_script = {key: valid}
            raise Exception("'{}' value ({}) is not a list. Please check the class member details: "
                            "\n\n{} ".format(key, value, self.output_definition(json_script)))

    def validate_other(self, key, value):
        """
        value definition is not a list or dict
        Note: not sure we have this case
        """
        valid = self.definition[key]

        if type(valid) != type(value):
            json_script = {key: valid}
            raise Exception("'{}' value ({}) is not the required type. Please check the class member details: "
                            "\n\n{} ".format(key, value, self.output_definition(json_script)))
