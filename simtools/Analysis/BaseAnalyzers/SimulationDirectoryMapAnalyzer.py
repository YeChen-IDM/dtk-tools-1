from simtools.Analysis.BaseAnalyzers import BaseAnalyzer


class SimulationDirectoryMapAnalyzer(BaseAnalyzer):
    """
    Analyzer allowing the user to retrieve data for each experiments containing:
    | Sim ID | Tags | Path |

    We do not need any select_simulation_data in this analyzer as all the data we need can be
    found in the simulation objects. Therefore only a finalize method is used.
    """

    def __init__(self):
        super().__init__(need_dir_map=True)

    def finalize(self, all_data: dict) -> dict:
        """
        Will go through every keys of the `all_data` parameters (the simulation objects) and construct the return
        dictionary by extracting the id, tags and physical path.

        Args:
            all_data: simulation -> selected data (here NONE)

        Returns:
            Dictionary associating experiment_id -> [{id:"", tag_1:"", tag_2:"", path:""}, {...}]

        """
        results = {}

        for simulation in all_data.keys():
            if simulation.experiment_id not in results:
                results[simulation.experiment_id] = []

            # Add the simulation to the results
            results[simulation.experiment_id].append({
                "id": simulation.id,
                **simulation.tags,
                "path": simulation.get_path()
            })

        return results

