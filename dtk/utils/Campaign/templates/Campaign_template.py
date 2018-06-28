###################################
# Our classes
###################################


class Campaign(BaseCampaign):
    _definition = {"Campaign_Name": "Empty Campaign", "Events": [], "Use_Defaults": {"default": True, "type": "bool"}}
    _validator = ClassValidator(_definition, 'Campaign')

    def __init__(self, Campaign_Name="Empty Campaign", Use_Defaults=True, Events=[], **kwargs):
        super(Campaign, self).__init__(**kwargs)
        self.Campaign_Name = Campaign_Name
        self.Use_Defaults = Use_Defaults
        self.Events = Events

    def add_campaign_event(self, ce):
        self.Events.append(ce)