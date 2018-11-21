from COMPS.Data import WorkItem
from COMPS.Data.WorkItem import WorkItemState

from simtools.Services.ObejctCatelog.ObjectInfoSvc import ObjectInfoSvc
from simtools.Utilities.COMPSUtilities import COMPS_login


class RunSvc:

    @staticmethod
    def run(item_id):
        info = ObjectInfoSvc.get_item_info(item_id)

        if info:
            endpoint = info["provider_info"].get("endpoint", None)
            if info['type'] == 'WI' and info['provider'] == 'COMPS':
                COMPS_login(endpoint)
                wi = WorkItem.get(item_id)
                if wi.state == WorkItemState.Created:
                    wi.commission()

