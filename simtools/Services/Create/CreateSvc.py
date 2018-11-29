import json
from COMPS.Data import WorkItem, WorkItemFile
from COMPS.Data.WorkItem import WorkerOrPluginKey, RelationType
from simtools.AssetManager.AssetCollection import AssetCollection
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
            asset_collection_id = kwargs['asset_collection_id']
            asset_files = kwargs['asset_files']
            user_files = kwargs['user_files']
            wo_kwargs = kwargs['wo_kwargs']
            command = kwargs['command']
            related_experiments = kwargs.get('related_experiments', None)

            # Collect asset files
            if not asset_collection_id:
                # Create a collection with everything that is in asset_files
                if len(asset_files.files) > 0:
                    ac = AssetCollection(local_files=asset_files)
                    ac.prepare("HPC")
                    asset_collection_id = ac.collection_id

            # Create a WorkItem
            wi = WorkItem(name=item_name, worker=WorkerOrPluginKey(item_type, plugin_key),
                          environment_name=comps_env, asset_collection_id=asset_collection_id)

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
            for af in user_files:
                wi_file = WorkItemFile(af.file_name, "input")
                wi.add_file(wi_file, af.absolute_path)

            # Save the work-item to the server
            wi.save()
            wi.refresh()

            # Sets the related experiments
            if related_experiments:
                for exp_id in related_experiments:
                    wi.add_related_experiment(exp_id, RelationType.DependsOn)

            return str(wi.id)
