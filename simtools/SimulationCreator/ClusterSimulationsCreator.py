import os
import shutil

from simtools.SimulationCreator.LocalSimulationCreator import LocalSimulationCreator


class ClusterSimulationCreator(LocalSimulationCreator):

    def add_files_to_simulation(self, s, cb):
        cb.dump_files(s.sim_dir)

        for f in cb.assets.master_collection.asset_files_to_use:
            if f.relative_path:
                os.makedirs(os.path.join(self.experiment.get_path(), "Assets", f.relative_path), exist_ok=True)

            shutil.copyfile(f.absolute_path, os.path.join(self.experiment.get_path(), "Assets", f.relative_path or "", f.file_name))

