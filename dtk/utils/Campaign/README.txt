##################################################
##  Feature related Notes for Campaign Classes  ##
##################################################


1. debug in command line:

   dtk generate_classes -e <exe path> -o <output path> --debug

   - check each of schemas

   Note #1: command won't create output folder if it doesn't exist. So make sure it exists!

   Note #2: both <exe path> and <output path> can't have space


2. command details: dtk generate_classes -e <exe path> -o <output path> --debug

   - step 1: run Eradication.exe
		to generate schema.json
   - step 2: run dtk_post_process_schema.py
		to generate schema_post.json
   - step 3: run transform_schema()
		to generat schema_parsed.json and schema_interventions.json
   - step 4: run build_campaign_enum_definitions()
		to generate CampaignEnum.py file
   - step 5: run build_campaign_class_definitions()
		to generate CampaignClass.py file


2. Campaign_Class_Generator.py

   - works as the same way as dtk generate_classes
   - allow user to run within dtk


4. User authnentication pop

   - from Eradication.exe
   - just ignore it


5. Enum construction

   <ClassName>_<PropName>_Enum

   ex: s = SimpleVaccine(Cost_To_Consumer=10, Dont_Allow_Duplicates=True,
               Vaccine_Type=SimpleVaccine_Vaccine_Type_Enum.Generic)


6. _class.to_json () takes two parameters

   def to_json(self, use_defaults=True, human_readability=True):
       ...


   Note: usually we can't garanteen json match!

   for ex: json1 --> class --> json2

       As json1 may contain some members with default values, however with use_defaults=True, we have removed all those members in json2 with the default values!!


7. Turn On/Off validation

   In dtk/utils/Campaign/utils/BaseCampaign.py

   # Turn on/off class members validation
   VALIDATE = True

   Note #1: Validation is On by default.

   Note #2: even with VALIDATE = True, we have by-passed all un-existing members!!


8. RawCampaignObject in interventions

    Only two existing interventions use this special class:

    - habitat_scale.py
    - intervention_states.py

    Note: this special class may also be used in DTKConfigBuilder.from_files method

          b = DTKConfigBuilder.from_files(config_name, campaign_name)


          if some class doesn't exist in CampaignClass.py, it will use RawCampaignObject to wrap json automatically


9. For un-existing member of a Campaign class, we take it and not raise exception

    Note: we commented out the code (rasie exception)


10. dtk_post_process_schema.py

    - can't just be replaced with new one
    - need to modify two or three lines (add param in method, save to different file)


11. Standalone scripts testing

   from CampaignClass import *

not

   from .CampaignClass import *


12. b = DTKConfigBuilder.from_defaults('MALARIA_SIM')

   - use System Class/Enum
   - user can clean up campaign evens if like
   - use can create new event based on standalone scripts and add it to campaign object within DTKConfigBuilder


13. b = DTKConfigBuilder.from_files(config_name, campaign_name)

   - use System Class/Enum
   - user can clean up campaign evens if like
   - use can create new event based on standalone scripts and add it to campaign object within DTKConfigBuilder


14. cb.add_event(...)

    # good and suggested way
    cb.add_event(RawCampaignObject({"custom1": "event1"}))

    # show warning
    cb.add_event({"custom2": "event2"})


15. import consideration


    # from .CampaignEnum import *

    try:
        from CampaignEnum import *
    except ImportError:
        from .CampaignEnum import *

    - Pycharm won't show red (unresolvable)


16. Auto conversion
    Scenario #1: json to class (or create class by hand)

    - if Enum type, auto convert string to enum
    - if bool type, auto convert int to bool


    ex: s = SimpleVaccine(Cost_To_Consumer=10, Dont_Allow_Duplicates=True,
               Vaccine_Type=SimpleVaccine_Vaccine_Type_Enum.Generic)

    ex: s = SimpleVaccine(Cost_To_Consumer=10, Dont_Allow_Duplicates=True,
               Vaccine_Type='Generic')

        Note: make sure 'Generic' -> SimpleVaccine_Vaccine_Type_Enum.Generic that SimpleVaccine_Vaccine_Type_Enum used in ClassValidator comes from standalone script CampaignEnum.py!!

	We create enum dynamically!!!


    Scenario #2: class to json

    - if Enum type, auto convert enum to string
    - if bool type, auto convert bool to 1/0


17. Class members validation

    - ClassValidator works with __setattr__ in BaseCampain class

      Note: can't change value in ClassValidator, has to do it in __setattr__ from BaseCampaign class

    - details...


18. Schema tranformation

    - need to double check and see if there is any special case we need to handle

    # original post-process schema
    schema_post.json

    # after our parsing and transformation
    schema_parsed.json
    schema_interventions.json

    [Handle List]
    	Case: special list, say 'Sim_Type', 'enum', 'default', 'Built-n'.
    	Case: len(value) == 1 and isinstance(value[0], dict)
    	Case: All items are dict

    [Handle Dict]
    	Case: <key>/<value> type
    	Case: Times/Values type
    	Case: 'base' in key
    	Case: 'base' not in key
		Contains 'type' and in ('int', 'float')
		Contains 'type' and "type": " idmType...."
                    - have to lookup, copy and transform...
		Contains 'type' and "type": not starts with "idmType"



19. CampaignClass.py generated based on standalone CampaignEnum.py (just generated right before generating CampaignClass.py)!

    Order:
    - generate CampaignEnum.py
    - generate CampaignClass.py

      Note: not use system CampaignEnum.py but the one just generated


20. Code Organization

Campaign
    Input
        schema.json
        schema_post.json
        schema_parsed.json
        schema_interventions.json
    templates
	Header_template.py
	Campaign_template.py
    utils
 	BaseCampaign.py
 	Campaign_Class_Generator.py
	CampaignDecoder.py
	CampaignEncoder.py
	CampaignManager.py
	ClassDecoder.py
	ClassParser.py
	dtk_post_process_schema.py
	RawCampaignObject.py
	SchemaDecoder.py
	SchemaParser.py



