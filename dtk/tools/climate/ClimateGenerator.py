import glob
import json
import logging
import os
import time
from typing import Tuple
from COMPS import Client
from COMPS.Data import AssetCollection
from COMPS.Data import QueryCriteria
from COMPS.Data import WorkItem, WorkItemFile
from COMPS.Data.WorkItem import WorkerOrPluginKey, WorkItemState
from simtools.COMPSAccess.InputDataWorker import InputDataWorker
from dtk.utils.ioformat.OutputMessage import OutputMessage as om
from simtools.SetupParser import SetupParser
from simtools.Utilities.COMPSUtilities import download_asset_collection
from simtools.Utilities.General import file_size
from logging import FileHandler
from logging import Formatter


logger = logging.getLogger(__name__)
logger_file_handler = FileHandler('ClimateGenerator_Log.log')
logger_file_handler.setLevel(logging.DEBUG)
logger_file_handler.setFormatter(Formatter('%(asctime)s: %(levelname)s: %(message)s'))
logger.addHandler(logger_file_handler)

# use 150 if you want resolution = 2.5arcmin
ALLOW_RESOLUTIONS = ['0', '150', '30']


class ClimateGenerator:
    def __init__(self, demographics_file_path: str, work_order_path: str, climate_files_output_path: str,
                 climate_project: str = "IDM-Zambia", start_year: str = str(2000), num_years: str = str(1),
                 resolution: str = str(0), project_root: str = 'v2017'):
        """
        Climate Generator handles generating climate related input files for the DTK from the COMPS Large DataSets

        Example:
            ```
            demographics_fp = os.path.join(path, 'demographics.json')
            output_path = os.path.dirname(os.path.abspath(__file__))
            climate_output_path = os.path.join(output_path, 'climate')
            cg = ClimateGenerator(demographics_file_path=demographics_fp,
                                  work_order_path=os.path.join(output_path, 'wo.json'),
                                  climate_files_output_path=climate_output_path,
                                  climate_project='IDM-Zambia',
                                  start_year='2008', num_years='1',
                                  resolution='0',
                                  project_root='v2017',
                                  idRef="Gridded world grump2.5arcmin")
            rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
            ```
        Args:
            demographics_file_path: The path to the Demographics file.
            work_order_path: Path to work order
            climate_files_output_path: Path to save climate output files and logs
            climate_project: Climate project name. See https://comps.idmod.org/#demoui for a List of supported
                             project names
            start_year: What year to start generation from
            num_years: Total years to generate climate files
            resolution: What resolution to run the climate generation at
            project_root:  What project root should we use. Usually the default is best as it reflects the latest data
                           sets
        """
        self.work_order_path = work_order_path
        self.demographics_file_path = demographics_file_path
        self.climate_project = climate_project
        self.start_year = start_year
        self.num_years = num_years
        if resolution not in ALLOW_RESOLUTIONS:
            raise ValueError(f"{resolution} is not a valid resolution. Please use one"
                             f" of the following values {', '.join(ALLOW_RESOLUTIONS)} ")
        self.resolution = resolution
        self.project_root = project_root
        self.wo = None

        self.validate_inputs()

        # Get the idRef from the demographics_file if notspecified
        # Raise FileNotFoundError if demographics_file_path does not exist.
        with open(demographics_file_path, 'r') as demographics_file:
            demog = json.load(demographics_file)
        demog_idref = demog['Metadata']['IdReference']

        # fetch idRef from the input demographics file's Metadata->IdReference field.
        self.idRef = demog_idref

        # use "wo.json" as the work order name and save it in the working directory if user inputs an empty string.
        if not self.work_order_path:
            self.work_order_path = "wo.json"
        work_order_dirname = os.path.dirname(self.work_order_path)
        if work_order_dirname:
            # Create folder to save wo.json if user input a path that doesn't exist
            if not os.path.isdir(work_order_dirname):
                os.mkdir(work_order_dirname)

        # save climate files to working directory if user passes in an empty string
        if not climate_files_output_path:
            climate_files_output_path = '.'
        self.climate_files_output_path = climate_files_output_path
        if not os.path.exists(self.climate_files_output_path): os.makedirs(self.climate_files_output_path)

    def validate_inputs(self):
        """
        Performs the following validations with the specified climate inputs

        1. Is their a project data set name(climate_project) exist in the Large Data API
        2. Is the specified year range valid for the specified project?

        This is called during construction of the Climate Generator object
        Returns: None

        """
        client = Client()
        client.login(SetupParser.get('server_endpoint'))
        response = client.get('/ld/DataSets.json')
        if response.status_code is not 200:
            raise EnvironmentError("Issue connecting to the COMPS Large Data Datasets API"
                                   f"({SetupParser.get('server_endpoint')}/api/ld/DataSets.json)")
        data_sets = response.json()
        try:
            project_data_set = [ds for ds in data_sets if ds['name'] == self.climate_project][0]
            end_year = int(self.start_year) + int(self.num_years)

            if project_data_set['endYear'] < end_year or int(self.start_year) < project_data_set['startYear']:
                raise ValueError(f"The Climate Project {self.climate_project} dataset does not support contain the "
                                 f"specified year range of {self.start_year} to {end_year}. The dataset supports the"
                                 f" years {project_data_set['startYear']} to {project_data_set['endYear']}.")
        except IndexError as e:  # The
            supported_projects = "\n".join(sorted([ds['name'] for ds in data_sets if ds['name'].startswith('IDM-')]))
            raise ValueError(f"Cannot locate the Climate Project with name: {self.climate_project}. The list of"
                             f" supported projects are as follows: {supported_projects}")

    @staticmethod
    def wait_for_work_item(wi: WorkItem, work_item_name: str='work item', check_state_interval: str=5) -> WorkItem.state:
        """
        Wait on the specified work item to either finish by Succeeding, Failing, or being Canceled
        Args:
            wi: Work item to wait on
            work_item_name: Name of work item to display as part of the periodic state check. Default to 'work item'
            check_state_interval: How often to check work items state in secs. Defaults to every 5

        Returns: Work item state

        """
        while wi.state not in (WorkItemState.Succeeded, WorkItemState.Failed, WorkItemState.Canceled):
            om(f'Waiting for {work_item_name} to complete (current state: {str(wi.state)}\n', style='flushed')
            time.sleep(check_state_interval)
            wi.refresh()
        return wi.state

    def create_work_item(self) -> WorkItem:
        """
        Creates the work item to generator the Climate Files

        Returns: work item

        """
        workerkey = WorkerOrPluginKey(name='InputDataWorker', version='1.0.0.0_RELEASE')
        wi = WorkItem('dtk-tools InputDataWorker WorkItem', workerkey, SetupParser.get('environment'))
        wi.set_tags({'dtk-tools': None, 'WorkItem type': 'InputDataWorker dtk-tools'})
        with open(self.work_order_path, 'rb') as workorder_file:
            # wi.AddWorkOrder(workorder_file.read())
            wi.add_work_order(data=workorder_file.read())
        with open(self.demographics_file_path, 'rb') as demog_file:
            wi.add_file(WorkItemFile(os.path.basename(self.demographics_file_path), 'Demographics', ''),
                        data=demog_file.read())
        wi.save()

        logger.info("Created request for climate files generation.")

        logger.info("Commissioning...")
        wi.commission()
        logger.info("CommissionRequested.")
        logger.info(f"Work item id is: {wi.id}.")

        return wi

    def download_climate_assets(self, wi: WorkItem) -> bool:
        """
        Download the climate asset files from the specified work item
        Args:
            wi:

        Returns: True if assests were located, otherwise False

        """

        # Get the collection with our files
        collections = wi.get_related_asset_collections()
        collection_id = collections[0].id
        comps_collection = AssetCollection.get(collection_id, query_criteria=QueryCriteria().select_children('assets'))

        # Get the files
        if len(comps_collection.assets) > 0:
            logger.info("Found output files:")
            for asset in comps_collection.assets:
                logger.info(f"- {asset.file_name} ({file_size(asset.length)})")

            logger.info(f"Downloading to {self.climate_files_output_path}...")
            # Download the collection
            download_asset_collection(comps_collection, self.climate_files_output_path)
            return True
        return False

    def generate_climate_files(self) -> Tuple[str, str, str]:
        """
        The main method of the class that performs the manages the workflow of generating climate files.

        The workflow is a follows
        1. The method first creates an InputDataWorker from the specified data inputs
        2. Then a work item is created from the InputDataWorker workorder
        3. Once the work item finishes, we check the work item state.
            If it's succeeded, we attempt to download the generated assets.
            If it's failed or canceled, we throw Error.
        4. If successful, we return the rain_file_name, temperature_file_name, humidity_file_name

        Returns:

        """
        # see InputDataWorker for other work options
        self.wo = InputDataWorker(demographics_file_path=self.demographics_file_path,
                                  wo_output_path=self.work_order_path,
                                  project_info=self.climate_project,
                                  start_year=str(self.start_year),
                                  num_years=str(self.num_years),
                                  resolution=str(self.resolution),
                                  idRef=self.idRef,
                                  project_root=self.project_root)

        # login to COMPS (if not already logged in) to submit climate files generation work order

        self.wo.wo_2_json()

        wi = self.create_work_item()

        # Wait until the work item is Done(Succeeded, Fails, or Canceled)
        wi_state = self.wait_for_work_item(wi)
        logger.info(f"Work item state is: {str(wi_state)}")

        if wi_state == WorkItemState.Succeeded:
            logger.info("Climate files SUCCESSFULLY generated")
            # try to download the climate files into climate_files_output_path if work item runs successfully.
            if self.download_climate_assets(wi):
                # return filenames; this use of re in conjunction w/ glob is not great; consider refactor
                rain_bin_re = os.path.abspath(f'{self.climate_files_output_path}/*rain*.bin')
                humidity_bin_re = os.path.abspath(f'{self.climate_files_output_path}/*humidity*.bin')
                temperature_bin_re = os.path.abspath(f'{self.climate_files_output_path}/*temperature*.bin')

                rain_file_name = os.path.basename(glob.glob(rain_bin_re)[0])
                humidity_file_name = os.path.basename(glob.glob(humidity_bin_re)[0])
                temperature_file_name = os.path.basename(glob.glob(temperature_bin_re)[0])

                logger.info('Climate files SUCCESSFULLY stored.')

                return rain_file_name, temperature_file_name, humidity_file_name

            else:
                logger.info('No output files found')
                raise ValueError("No Climate output files generated. Please check work item log")
        else:
            # raise ValueError if work item doesn't run successfully.
            logger.info('Work item status is not Succeeded')
            raise ValueError("Work item status is not Succeeded. Please check work item log")

