import json

from COMPS.Data import WorkItem, WorkItemFile
from COMPS.Data.WorkItem import WorkerOrPluginKey

from simtools.Utilities.COMPSUtilities import COMPS_login


class CreateSvc:

    @staticmethod
    def create(provider, provider_info, type, **kwargs):
        endpoint = provider_info.get("endpoint", None)
        if type == 'WI' and provider == 'COMPS':
            COMPS_login(endpoint)
            # collect variables
            item_name = kwargs['item_name']
            item_type = kwargs['item_type']
            docker_image = kwargs['docker_image']
            plugin_key = kwargs['plugin_key']
            comps_env = kwargs['comps_env']
            tags = kwargs['tags']
            files = kwargs['files']
            wo_kwargs = kwargs['wo_kwargs']
            command = kwargs['command']

            # Create a WorkItem
            wi = WorkItem(item_name, WorkerOrPluginKey(item_type, plugin_key), comps_env)

            # set tags
            if tags:
                wi.set_tags(tags)

            # Add work order file
            wo = {
                "WorkItem_Type": item_type,
                "Execution": {
                    "ImageName": docker_image,
                    "Command": command
                }
            }
            wo.update(wo_kwargs)
            wi.add_work_order(data=json.dumps(wo).encode('utf-8'))

            # Add additional files
            for af in files:
                wi_file = WorkItemFile(af.file_name, "input")
                wi.add_file(wi_file, af.absolute_path)

            # Save the work-item to the server
            wi.save()
            wi.refresh()

            return str(wi.id)
