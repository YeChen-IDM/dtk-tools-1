import time

from COMPS.Data.WorkItem import WorkItemState

from simtools.AssetManager.FileList import FileList
from simtools.Services.Create.CreateSvc import CreateSvc
from simtools.Services.ObejctCatelog.ObjectInfoSvc import ObjectInfoSvc
from simtools.Services.Run.RunSvc import RunSvc
from simtools.Services.Status.StatusSvc import StatusSvc
from simtools.SetupParser import SetupParser
from simtools.Utilities.COMPSUtilities import COMPS_login


class WorkItemManager:

    def __init__(self, item_name="DockerWorker WorkItem", item_type="DockerWorker",
                 docker_image="ubuntu1804python3dtk", provider="COMPS", type="WI", plugin_key="1.0.0.0_RELEASE",
                 command=None, user_files=FileList(), tags=None, wo_kwargs=None, related_experiments=None):

        self.item_name = item_name
        self.item_id = None
        self.provider = provider
        self.tags = tags or {}
        self.comps_host = SetupParser.get('server_endpoint')
        self.comps_env = SetupParser.get('environment')
        self.item_type = item_type
        self.plugin_key = plugin_key
        self.docker_image = docker_image
        self.files = user_files
        self.wo_kwargs = wo_kwargs or {}
        self.command = command
        self.type = type
        self.related_experiments = related_experiments

    def execute(self, check_status=True):

        self.create()

        self.run()

        self.wait_for_finish(check_status)

    def create(self):
        # Login
        COMPS_login(self.comps_host)

        # Create a WorkItem
        provider_info = {"endpoint": self.comps_host, "environment": self.comps_env}
        kwargs = {"item_name": self.item_name, "item_type": self.item_type, "docker_image": self.docker_image,
                  "plugin_key": self.plugin_key, "comps_env": self.comps_env, "tags": self.tags, "files": self.files,
                  "wo_kwargs": self.wo_kwargs, "command": self.command, "related_experiments": self.related_experiments}
        self.item_id = CreateSvc.create(self.provider, provider_info, self.type, **kwargs)

        # Refresh local object db
        ObjectInfoSvc.create_item(type=self.type, provider=self.provider, provider_info=provider_info,
                                  item_id=str(self.item_id))

    def run(self):
        RunSvc.run(self.item_id)

    def status(self):
        return StatusSvc.get_status(self.item_id)

    def wait_for_finish(self, check_status=True, timeout=3600):
        if check_status:
            start = time.clock()
            state = self.status()
            states = [WorkItemState.Succeeded, WorkItemState.Failed, WorkItemState.Canceled]
            while state not in states and time.clock() - start < timeout:
                time.sleep(5)
                state = self.status()
                print('State -> {} '.format(state.name))
        else:
            print('WorkItem created in {}.'.format(self.provider))

    def add_file(self, af):
        self.files.add_asset_file(af)

    def add_wo_arg(self, name, value):
        self.wo_kwargs[name] = value

    def clear_user_files(self):
        self.files = FileList()

    def clear_wo_args(self):
        self.wo_kwargs = {}
