from uuid import UUID


class COMPSCache:
    _simulations = {}
    _experiments = {}
    _collections = {}

    @classmethod
    def simulation(cls, sid, criteria=None, children=None):
        if criteria:
            return cls.query_simulation(sid, criteria, children)

        if sid not in cls._simulations:
            cls.load_simulation(sid, criteria, children)

        return cls._simulations[sid]

    @classmethod
    def experiment(cls, eid, criteria=None, children=None):
        if criteria:
            return cls.query_experiment(eid, criteria, children)

        if eid not in cls._experiments:
            cls.load_experiment(eid, criteria, children)

        return cls._experiments[eid]

    @classmethod
    def collection(cls, cid):
        if cid not in cls._collections:
            cls.load_collection(cid)

        return cls._collections[cid]

    @classmethod
    def add_experiment_to_cache(cls, e):
        cls._experiments[str(e.id)] = e

    @classmethod
    def add_simulation_to_cache(cls, s):
        cls._simulations[str(s.id)] = s

    @classmethod
    def load_collection(cls, cid):
        try:
            UUID(cid)
            collection = cls.query_collection(cid)
        except ValueError:
            collection = cls.query_collection(cname=cid)
        cls._collections[cid] = collection

    @classmethod
    def load_simulation(cls, sid, criteria=None, children=None, force=False):
        if sid in cls._simulations and not force:
            return

        s = cls.query_simulation(sid, criteria, children)
        cls._simulations[sid] = s
        cls.load_experiment(s.experiment_id)

    @classmethod
    def load_experiment(cls, eid, criteria=None, children=None, force=False):
        if eid in cls._experiments and not force:
            return

        e = cls.query_experiment(eid, criteria, children)
        cls._experiments[eid] = e

        for s in cls.get_experiment_simulations(e):
            cls._simulations[str(s.id)] = s

    @staticmethod
    def get_experiment_simulations(exp):
        return exp.get_simulations()

    @staticmethod
    def query_collection(cid=None, cname=None, criteria=None):
        from COMPS.Data import QueryCriteria
        from COMPS.Data import AssetCollection
        criteria = criteria or QueryCriteria().select_children('assets')
        if cid:
            return AssetCollection.get(id=cid, query_criteria=criteria)

        criteria.where_tag('Name={}'.format(cname))
        results = AssetCollection.get(query_criteria=criteria)
        if len(results) >= 1: return results[0]

    @staticmethod
    def query_experiment(eid=None, criteria=None, children=None):
        from COMPS.Data import Experiment
        from COMPS.Data import QueryCriteria

        criteria = criteria or QueryCriteria()
        children = children or ["tags"]
        criteria.select_children(children)

        exp = Experiment.get(eid, query_criteria=criteria)
        return exp

    @staticmethod
    def query_simulation(sid, criteria=None, children=None):
        from COMPS.Data import Simulation
        from COMPS.Data import QueryCriteria
        if children:
            criteria = criteria or QueryCriteria()
            criteria.select_children(children)

        return Simulation.get(sid, query_criteria=criteria)
