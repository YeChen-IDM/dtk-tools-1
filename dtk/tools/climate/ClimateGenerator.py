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

logger = logging.getLogger(__name__)
logging.basicConfig(filename='ClimateGenerator_Log.log', level=logging.DEBUG)

ALLOW_RESOLUTIONS = ['0', '2.5', '30']


class ClimateGenerator:
    def __init__(self, demographics_file_path: str, work_order_path: str, climate_files_output_path: str,
                 climate_project: str = "IDM-Zambia", start_year: str = str(2000), num_years: str = str(1),
                 resolution: str = str(0), idRef: str = None, project_root: str = 'v2017'):
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


        :param demographics_file_path: The path to the Demographics file.
        :param work_order_path: Path to work order
        :param climate_files_output_path:  Path to save climate output files and logs
        :param climate_project: Climate project name. See https://comps.idmod.org/#demoui for
        a List of supported project names
        :param start_year: What year to start generation from
        :param num_years: Total years to
        :param resolution: What resolution to run the climate generation at
        :param idRef: Demographics file IdRefernce. Optional, if not specified we attempt to fetch it from the
            input demographics file's Metadata->IdReference field.
        :param project_root: What project root should we use. Usually the default is best as it reflects the
            latest data sets
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
        demog = json.load(open(demographics_file_path, 'r'))
        demog_idref = demog['Metadata']['IdReference']
        if not idRef:
            self.idRef = demog_idref
        else:
            self.idRef = idRef
            if idRef != demog_idref:
                logger.info(f"/!\\ Warning: the idref of the demographics file ({demog_idref}) "
                            f"is different from the one passed ({idRef})")

        self.climate_files_output_path = climate_files_output_path
        if not os.path.exists(self.climate_files_output_path): os.makedirs(self.climate_files_output_path)

    def validate_inputs(self):
        """
        Performs the following validations with the specified climate inputs

        1. Is their a project data set name(climate_project) exist in the Large Data API
        2. Is the specified year range valid for the specified project?

        This is called during construction of the Climate Generator object
        :return: None
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
    def wait_for_work_item(wi, work_item_name='work item', check_state_interval=5):
        """
        Wait on the specified work item to either finish by Succeeding, Failing, or being Canceled

        :param wi: Work item to wait on
        :param work_item_name: Name of work item to display as part of the periodic state check. Default to 'work item'
        :param check_state_interval: How often to check work items state in secs. Defaults to every 5
        :return: None
        """
        while wi.state not in (WorkItemState.Succeeded, WorkItemState.Failed, WorkItemState.Canceled):
            om(f'Waiting for {work_item_name} to complete (current state: {str(wi.state)}', style='flushed')
            time.sleep(check_state_interval)
            wi.refresh()

    def create_work_item(self) -> WorkItem:
        """
        Creates the work item to generator the Climate Files

        :return: work item
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
        return wi

    def download_climate_assets(self, wi: WorkItem) -> bool:
        """
        Download the climate asset files from the specified work item
        :param wi:
        :return: True if assests were located, otherwise False
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

            logger.info(f"\nDownloading to {self.climate_files_output_path}...")
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
        3. Once the work item finishes, we attempt to download the generated assets
        4. If successful, we return the temperature_file_name, humidity_file_name

        :return:
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
        self.wait_for_work_item(wi)
        logger.info("Climate files SUCCESSFULLY generated")

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
