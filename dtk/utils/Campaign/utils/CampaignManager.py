import os
import json
from enum import Enum
import importlib
import inspect
import subprocess
import shutil

import astor
import astunparse
from ast import *

from dtk.utils.Campaign.utils.ClassParser import ClassParser
from dtk.utils.Campaign.utils.SchemaParser import SchemaParser
from dtk.utils.Campaign.utils.dtk_post_process_schema import application


class CampaignManager(object):
    """
    Main utility object to help
      - parse and transform Eradication schema
      - generate Campaign classes/enums
      - support command: dtk generate_classes
    """

    dir_path = os.path.dirname(__file__)
    CAMPAIGN_ROOT = os.path.dirname(dir_path)
    INPUT_ROOT = os.path.join(CAMPAIGN_ROOT, 'input')
    TEMPLATE_ROOT = os.path.join(CAMPAIGN_ROOT, 'templates')


    @classmethod
    def execute_all(cls):
        # step 1: Parse Schema and transform it to a 'standard' format
        cls.transform_schema()

        # step 2: Create CampaignEnum.py file
        cls.build_campaign_enum_definitions()

        # step 3: create CampaignClass.py file
        cls.build_campaign_class_definitions()


    #######################################
    # Generate Campaign Classes directly
    #     from Eradication Schema
    #######################################

    @classmethod
    def generate_campaign_classes(cls, exe_file, out_folder, debug=False):

        # Step #1: generate schema
        OUT_FOLDER = out_folder
        SCHEMA_FILE = "schema.json"
        command = '"{}" --get-schema --schema-path "{}"'.format(exe_file, os.path.join(OUT_FOLDER, SCHEMA_FILE))

        # temp files
        files = [os.path.join(OUT_FOLDER, file_name) for file_name in
                 ["StdOut.txt", "StdErr.txt", "schema.json", "schema_post.json", "schema_parsed.json",
                  "schema_interventions.json", "__pycache__"]]

        try:
            with open(os.path.join(OUT_FOLDER, "StdOut.txt"), "w") as out, open(os.path.join(OUT_FOLDER, "StdErr.txt"),
                                                                                "w") as err:
                # Launch the command
                # p = subprocess.Popen(command, cwd=OUT_FOLDER, shell=False, stdout=out, stderr=err)    # working!
                p = subprocess.run(command, cwd=OUT_FOLDER, shell=False, stdout=out, stderr=err)  # working!
        except Exception as ex:
            print("Error encountered while processing schema.")
            for file_path in files:
                cls.remove_file(file_path)
            raise ex

        # Step #2: generate post-process schema
        application(os.path.join(OUT_FOLDER, SCHEMA_FILE), OUT_FOLDER)

        # Step #3: parse the post-process schema, generate CampaignClass.py and CampaignEnum.py
        cls.INPUT_ROOT = OUT_FOLDER
        cls.CAMPAIGN_ROOT = OUT_FOLDER
        cls.TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), '..', 'templates')

        try:
            # Transform schema, generate CampaignClass.py and CampaignEnum.py
            cls.execute_all()
        except Exception as ex:
            print("Error encountered while generating classes.")
            raise ex
        finally:
            if not debug:
                for file_path in files:
                    cls.remove_file(file_path)

    @classmethod
    def remove_file(cls, file_path):
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


    ###############################
    # Transform Eradication Schema
    ###############################

    @classmethod
    def transform_schema(cls):
        schema_parser = SchemaParser(os.path.join(cls.INPUT_ROOT, r'schema_post.json'))
        schema_output = schema_parser.parse(None)

        # save schema_output
        schema_parser.save_schema_to_file(os.path.join(cls.INPUT_ROOT, 'schema_parsed'),
                                          json.dumps(schema_output, indent=3))

        # only interested in interventions
        if 'idmTypes' in schema_output:
            INTERVENTION_CLASSES = {"interventions": schema_output['idmTypes']}
            schema_parser.save_schema_to_file(os.path.join(cls.INPUT_ROOT, 'schema_interventions'),
                                              json.dumps(INTERVENTION_CLASSES, indent=3))


    ###############################
    # Build Campaign Enums
    ###############################

    @classmethod
    def build_campaign_enum_definitions(cls):
        class_parser = ClassParser(os.path.join(cls.INPUT_ROOT, r'schema_interventions.json'))
        class_parser.parse(None)

        d = ["from enum import Enum, auto\n"]
        for enum_name, args_list in class_parser.class_decoder.enum_dict.items():
            enum_definition = cls.build_enum_definition("{}_Enum".format(enum_name), args_list)
            d.append(enum_definition)

        f = open(os.path.join(cls.CAMPAIGN_ROOT, 'CampaignEnum.py'), 'w')
        f.write('\n\n'.join(d))
        f.close()

    @classmethod
    def build_enum_definition(cls, enum_name, args_list):
        enum_ast = cls.build_enum_ast(enum_name, args_list)
        enum_definition = astor.to_source(enum_ast)

        return enum_definition

    @classmethod
    def build_enum_ast(cls, enum_name, args_list):
        _assigns = []
        for (i, v) in enumerate(args_list):
            a = Assign(
                  targets=[Name(
                    id=v,
                    ctx=Store())],
                  value=Call(
                    func=Name(
                      id='auto',
                      ctx=Load()),
                    args=[],
                    keywords=[]))
            _assigns.append(a)

        enum_ast = Module(body=[ClassDef(
              name=enum_name,
              bases=[Name(
                id='Enum',
                ctx=Load())],
              keywords=[],
              body=_assigns,
              decorator_list=[])])

        return enum_ast

    ###############################
    # Build Campaign Classes
    ###############################

    @classmethod
    def build_campaign_class_definitions(cls):

        # dynamically load CampaignEnum.py
        module_file_path = os.path.join(cls.CAMPAIGN_ROOT, "CampaignEnum.py")
        module_spec = importlib.util.spec_from_file_location(
            'CampaignEnum', module_file_path)
        imported_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(imported_module)

        # make all enum in CampaignEnum.py available in globals
        enum_list = inspect.getmembers(imported_module, inspect.isclass)
        enum_dict = {k: v for k, v in enum_list}
        globals().update(enum_dict)

        # parse for all class details
        class_parser = ClassParser(os.path.join(cls.INPUT_ROOT, r'schema_interventions.json'))
        class_parser.parse()

        # generate class definition one by one
        d = [cls.copy_header_template()]
        base_name = "BaseCampaign"
        for cls_name, params in class_parser.class_decoder.class_params.items():
            args_list = params['args']
            defaults_list = params['defaults']
            definition = params['_definition']

            if 'self' not in args_list:
                args_list.insert(0, 'self')

            # build ast
            class_ast = cls.build_class_ast(cls_name, base_name, args_list, defaults_list, definition)

            # generate class definition source code
            # class_definition = astor.to_source(class_ast)         # single quote dict with certain format
            class_definition = astunparse.unparse(class_ast)        # single quote one line dict

            d.append(class_definition)

        d.append(cls.copy_campaign_template())

        # save all class definitions to a file
        f = open(os.path.join(cls.CAMPAIGN_ROOT, 'CampaignClass.py'), 'w')
        f.write('\n'.join(d))
        f.close()

    @classmethod
    def copy_header_template(cls):
        path = os.path.join(cls.TEMPLATE_ROOT, 'Header_template.py')
        return open(path, 'r').read()

    @classmethod
    def copy_campaign_template(cls):
        path = os.path.join(cls.TEMPLATE_ROOT, 'Campaign_template.py')
        return open(path, 'r').read()

    @classmethod
    def build_class_ast(cls, cls_name, base_name, args_list, defaults_list, definition=None):
        _args, _defaults, _assigns, _definition = cls.generate_ast_attributes(cls_name, args_list, defaults_list, definition)

        class_ast = ClassDef(
          name=cls_name,
          bases=[Name(
            id=base_name,
            ctx=Load())],
          keywords=[],
          body=[
            Assign(
              targets=[Name(
                id='_definition',
                ctx=Store())],
              value=_definition
            ),
            Assign(
              targets=[Name(
                  id='_validator',
                  ctx=Store())],
              value=Call(
                  func=Name(
                      id='ClassValidator',
                      ctx=Load()),
                  args=[
                      Name(
                          id='_definition',
                          ctx=Load()),
                      Str(s=cls_name)],
                  keywords=[])
            ),
            FunctionDef(
              name='__init__',
              args=arguments(
                  args=_args,
                  vararg=None,
                  kwonlyargs=[],
                  kw_defaults=[],
                  kwarg=arg(
                      arg='kwargs',
                      annotation=None),
                  defaults=_defaults),
              body=[
                   Expr(value=Call(
                       func=Attribute(
                           value=Call(
                               func=Name(
                                   id='super',
                                   ctx=Load()),
                               args=[
                                   Name(
                                       id=cls_name,
                                       ctx=Load()),
                                   Name(
                                       id='self',
                                       ctx=Load())],
                               keywords=[]),
                           attr='__init__',
                           ctx=Load()),
                       args=[],
                       keywords=[keyword(
                           arg=None,
                           value=Name(
                               id='kwargs',
                               ctx=Load()))]))
                   ] + _assigns,
              decorator_list=[],
              returns=None)
          ],
          decorator_list=[]
        )

        return class_ast

    @classmethod
    def generate_ast_attributes(cls, cls_name, args_list, defaults_list, definition=None):

        def adjust_default_for_enum():
            if definition is None:
                return

            adjust = {}
            for i in range(0, len(defaults_list)):
                key = args_list[i + 1]
                val = defaults_list[i]      # note: same as value['default']

                value = definition[key]
                if isinstance(value, dict) and value.get('type', None) == 'enum':
                    default = value['default']
                    d_enum = globals()["{}_{}_Enum".format(cls_name, key)]
                    adjust[str(i)] = d_enum[default]

            for i, v in adjust.items():
                defaults_list[int(i)] = v

        def adjust_default_for_bool():
            if definition is None:
                return

            adjust = {}
            for i in range(0, len(defaults_list)):
                key = args_list[i + 1]
                val = defaults_list[i]      # note: same as value['default']

                value = definition[key]
                if isinstance(value, dict) and value.get('type', None) == 'bool':
                    default = value['default']
                    adjust[str(i)] = False if default == 0 else True

            for i, v in adjust.items():
                defaults_list[int(i)] = v

        def transform(v):
            if isinstance(v, bool):
                return NameConstant(value=v)
            elif isinstance(v, int):
                return Num(n=v)
            elif isinstance(v, float):
                return Num(n=v)
            elif isinstance(v, str):
                return Str(s=v)
            elif isinstance(v, Enum):
                obj = Attribute(
                      value=Name(
                        id=v.__class__.__name__,
                        ctx=Load()),
                      attr=v.name,
                      ctx=Load()
                )
                return obj
            elif v is None:
                return NameConstant(value=v)
            elif isinstance(v, list):
                obj = List(
                      elts=[transform(i) for i in v],
                      ctx=Load()
                )
                return obj
            elif isinstance(v, dict):
                obj = Dict(
                      keys=[transform(i) for i in v.keys()],
                      values=[transform(i) for i in v.values()]
                )
                return obj
            else:
                return v

        _args = []
        _defaults = []
        _assigns = []
        _definition = None

        # take enum into consideration
        adjust_default_for_enum()

        # take bool into consideration
        adjust_default_for_bool()

        for s in args_list:
            _args.append(arg(arg=s, annotation=None))

            if s != 'self':
                a = Assign(
                    targets=[Attribute(
                      value=Name(
                        id='self',
                        ctx=Load()),
                      attr=s,
                      ctx=Store())],
                    value=Name(
                      id=s,
                      ctx=Load())
                )
                _assigns.append(a)

        for v in defaults_list:
            _defaults.append(transform(v))

        if definition:
            _definition = transform(definition)

        if definition is None:
            return _args, _defaults, _assigns

        return _args, _defaults, _assigns, _definition


    ####################################################
    # Transform Campaign JSON file to Campaign Classes
    ####################################################

    @classmethod
    def json_file_to_classes(cls, campaign_file):
        try:
            json_data = json.load(open(campaign_file, 'r'))
        except IOError:
            raise Exception('Unable to read file: %s' % campaign_file)

        return cls.json_to_classes(json_data)


    @classmethod
    def json_to_classes(cls, json_data):
        from dtk.utils.Campaign.utils.CampaignDecoder import CampaignDecoder
        # add class so that we can use JSONDecoder to transform it to a Campaign Class
        if 'class' not in json_data:
            json_data['class'] = 'Campaign'

        # transform Campaign (JSON) to Campaign (Class)
        json_str = json.dumps(json_data, indent=3)
        campaign = json.loads(json_str, cls=CampaignDecoder)

        return campaign



if __name__ == "__main__":
    """
        three thing in one:
        1. transform schema; 
        2. generate CampaignEnum.py; 
        3. generate CampaignClass.py
    """

    CampaignManager.execute_all()

    print('Successfully executed all!')
