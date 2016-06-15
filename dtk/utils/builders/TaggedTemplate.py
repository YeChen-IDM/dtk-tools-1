from dtk.utils.parsers.JSON import json2dict
import os
import logging

import dtk.utils.builders.BaseTemplate as BaseTemplate

logger = logging.getLogger(__name__)

class TaggedTemplate(BaseTemplate.BaseTemplate):
    """
    A class for building, modifying, and writing input files marked with tags (e.g. __KP), including campaign,
    demographics, and potentially even config files.

    The idea here is that if you have a json file with deep nesting or a long array of events, you can plant "tags" within the file to facilitate parameter reference.  Tags are new keys adjacent to existing keys within the document.  The first part of the tagged parameter must exactly match the original parameter you wish to change.  The comes the tag, e.g. __KP.  Any string can follow the tag to uniquely identify it.

        For example, the tagged demographic coverage parameter, Demographic_Coverage__KP_Second_Coverage, in the example below allows you to set the Demographic_Coverage parameter of the second event only.

        {
            "Events" : [
                {
                    "Name": "First",
                    "Demographic_Coverage": 0.25,
                    "Range__KP_First": "<-- MARKER (this string is arbitrary)",
                    "Range" {
                        "Min": 0,
                        "Max": 1,
                    }
                },
                {
                    "Name": "Second",
                    "Demographic_Coverage__KP_Second_Coverage": "<-- MARKER (this string is arbitrary)",
                    "Demographic_Coverage": 0.25
                }
            ]
        }

        You can do some neat things with tags.
        * You can place a tagged parameter, e.g. Demographic_Coverage__KP_Second_Coverage, in several places.  The value will be set everywhere the tagged parameter is found.  For now, the whole tagged parameter must match, so Something_Else__KP_Second_Coverage would not receive the same value on set_param.
        * You can reference relateive to the tagged parameters, e.g. Range__KP_First.Min = 3
        * You don't have to use __KP, just set the tag parameter in the constructor.

    """

    def __init__(self, filename, contents, tag='__KP'):
        """
        Initialize a TaggedTemplate.

        :param filename: The name of the template file.  This is not the full path to the file, just the filename.
        :param contents: The contents of the template file
        :param tag: The string that marks parameters in the file.  Note, this must begin with two underscores.
        """

        if tag[:2] != '__':
            logger.error('Tags must start with two underscores, __.  The tag you supplied (' + tag + ') did not meet this criteria.')

        super(TaggedTemplate, self).__init__(filename, contents)

        self.tag = tag
        self.tag_dict = self.__findKeyPaths(self.contents, self.tag)


    @classmethod
    def from_file(cls, template_filepath, tag='__KP'):
        # Read in template
        logger.info( "Reading template from file:" + template_filepath )
        content = json2dict(template_filepath)

        # Get the filename and create a TaggedTemplate
        template_filename = os.path.basename(template_filepath)

        return cls(template_filename, content, tag)


    # ITemplate functions follow
    def get_param(self, param):
        """
        Get param value(s).
        :return: a tuple of values and paths
        """
        values = []
        expanded_params = self.expand_tag(param)
        for expanded_param in expanded_params:
            #value = self.__get_expanded_param(expanded_param)
            (_,value) = super(TaggedTemplate, self).get_param(expanded_param)
            values.append(value)

        return (expanded_params, values)


    def set_param(self, param, value):
        """
        Call set_param to set a parameter in the tagged template file.

        This function forst expands the tagged param to a list of full addresses, and then sets the value at each address.

        :param param: The parameter to set, e.g. CAMPAIGN.My_Parameter__KP_Seeding.Max
        :param value: The value to place at expanded parameter loci.
        :return: Simulation tags
        """
        sim_tags = {}
        for expanded_param in self.expand_tag(param):
            tag = super(TaggedTemplate, self).set_param(expanded_param, value)
            assert(len(tag)==1)
            sim_tags[tag.keys()[0]] = tag.values()[0]
            sim_tags["[BUILDER] "+param] = value

        return sim_tags


    def expand_tag(self, param):
        expanded_params = []

        if '.' in param:
            tokens = param.split('.')
            first_tok = tokens[0]
        else:
            tokens = []
            first_tok = param

        if self.tag in first_tok:
            key = self.__extractKey(first_tok)
            return ['.'.join( [path]+tokens[1:]) for path in self.tag_dict[key] ]

        return []


    def __findKeyPaths(self, search_obj, key_fragment, partial_path=[]):
        """
        Builds a dictionary of results from recurseKeyPaths.
        """
        paths_found =  self.__recurseKeyPaths(search_obj, key_fragment, partial_path)

        path_dict = {}
        for path in paths_found:
            # The last part of the path should contain the key_fragment (__KP)
            # Take as the key everything after the tag
            key = self.__extractKey(path[-1])

            # Truncate from key_fragment in k (lop off __KP_etc)
            path[-1] = path[-1].split(key_fragment)[0]

            path_str = '.'.join( str(p) for p in path)

            path_dict.setdefault(key,[]).append(path_str)

        return path_dict


    def __extractKey(self, string):
        index = string.find(self.tag)
        if index < 0:
            raise Exception( "[%s] Failed to find key fragment %s in string %s.", self.get_filename(), self.tag, string)
        return string[index + len(self.tag):]


    def __recurseKeyPaths(self, search_obj, key_fragment, partial_path=[]):
        """
        Locates all occurrences of keys containing key_fragment in json-object 'search_obj'.
        """
        paths_found = []

        if '.' in key_fragment:
            tmp = key_fragment.split('.')
            key_fragment = tmp[0]

        if isinstance(search_obj, dict):
            for k,v in search_obj.iteritems():
                if key_fragment in k:
                    paths_found.append( partial_path + [ k ])

            for k in search_obj.iterkeys():
                paths = self.__recurseKeyPaths(search_obj[k], key_fragment, partial_path + [k])
                for p in paths:
                    paths_found.append(p)

        elif isinstance(search_obj, list):
            for i in range(len(search_obj)):
                paths = self.__recurseKeyPaths(search_obj[i], key_fragment, partial_path + [i])
                for p in paths:
                    paths_found.append(p)

        return paths_found


class CampaignTemplate(TaggedTemplate):
    def set_params_and_modify_cb(self, params, cb):
        tags = self.set_params(params)

        cb.set_param('Campaign_Filename', self.get_filename())
        cb.campaign = self.get_contents()

        return tags


class DemographicsTemplate(TaggedTemplate):
    def set_params_and_modify_cb(self, params, cb):
        demographics_filenames = cb.params['Demographics_Filenames']
        # Make sure the filename is listed in Demographics_Filenames
        if self.get_filename() not in demographics_filenames:
            raise Exception( "Using template with filename %s for demographics, but this filename is not included in Demographics_Filenames: %s", self.get_filename(), demographics_filenames)

        self.set_params(params)
        cb.add_input_file(self.filename.replace(".json",""), self.get_contents())

