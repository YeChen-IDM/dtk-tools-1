from models.python.core.PythonConfigBuilder import PythonConfigBuilder
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.SetupParser import SetupParser

if __name__ == "__main__":
    SetupParser.init("HPC")
    pb = PythonConfigBuilder(python_file=r"..\inputs\my_model.py")
    em = ExperimentManagerFactory.from_cb(pb)
    em.run_simulations(exp_name="Test python")
