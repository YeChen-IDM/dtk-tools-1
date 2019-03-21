import unittest
import importlib

class TestInstallation(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simtools_ALL_import(self):
        import simtools
        self.assertIsNotNone(simtools)

    def test_simtools_analysis_spec(self):
        module = "simtools.Analysis"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)
#<editor-fold desc='simtools.Analysis space'>

    def test_simtools_analysis_OutputParser_import(self):
        expected_methods = ["load_bin_file","load_csv_file","load_json_file",
                            "load_raw_file","load_txt_file","load_xlsx_file","parse"]
        from simtools.Analysis.OutputParser import SimulationOutputParser as test
        self.report_mising_methods("SimulationOutputParser", expected_methods, test)

    def test_simtools_analysis_DataRetrievalProcess_import(self):
        expected_methods = ['get_asset_files_for_simulation_id', 'itertools', 'os', 'retrieve_COMPS_AM_files',
                            'retrieve_SSMT_files', 'retrieve_data', 'retrieve_data_for_simulation', 'retry',
                            'set_exception', 'traceback']
        from simtools.Analysis import DataRetrievalProcess as test
        self.report_mising_methods("DataRetrievalProcess", expected_methods, test)

    def test_simtools_analysis_AnalyzeManager_import(self):
        expected_methods = ['animation', 'collections', 'init_logging', 'logger', 'on_off', 'os', 'pluralize',
                            'pool_worker_initializer', 're', 'retrieve_data', 'retrieve_experiment',
                            'retrieve_simulation', 'sys', 'time', 'verbose_timedelta']
        from simtools.Analysis import AnalyzeManager as test
        self.report_mising_methods("AnalyzeManager", expected_methods, test)

    def test_simtools_analysis_AnalyzeHelper_import(self):
        expected_methods = ['analyze', 'check_existing_batch', 'check_status', 'clean_batch', 'clear_batch',
                            'collect_analyzers', 'collect_experiments_simulations', 'collect_simulations',
                            'compare_two_ids_list', 'consolidate_experiments_with_options', 'create_batch',
                            'delete_batch', 'init_logging', 'list_batch', 'load_config_module', 'logger', 'os',
                            'retrieve_experiment', 'retrieve_item', 'retrieve_simulation', 'save_batch']
        from simtools.Analysis import AnalyzeHelper as test
        self.report_mising_methods("AnalyzeHelper", expected_methods, test)

    def test_simtools_analysis_baseanalyzers_BaseAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import BaseAnalyzer as test
        expected_methods =['destroy', 'filter', 'finalize', 'initialize', 'per_experiment', 'select_simulation_data']
        self.report_mising_methods("BaseAnalyzer", expected_methods, test)
        pass

    def test_simtools_analysis_baseanalyzers_BaseCacheAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import BaseCacheAnalyzer as test
        expected_methods = ['destroy', 'filter', 'finalize', 'from_cache', 'initialize', 'is_in_cache', 'keys',
                            'per_experiment', 'select_simulation_data', 'to_cache']
        self.report_mising_methods("BaseAnalyzer", expected_methods, test)
        pass

    def test_simtools_analysis_baseanalyzers_BaseCalibrationAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import BaseCalibrationAnalyzer as test
        expected_methods = ['cache', 'destroy', 'filter', 'finalize', 'initialize', 'per_experiment',
                            'select_simulation_data']
        self.report_mising_methods("BaseAnalyzer", expected_methods, test)
        pass

    def test_simtools_analysis_baseanalyzers_DownloadAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import DownloadAnalyzer as test
        expected_methods = ['destroy', 'filter', 'finalize', 'get_sim_folder', 'initialize', 'per_experiment',
                            'select_simulation_data']
        self.report_mising_methods("BaseAnalyzer", expected_methods, test)
        pass

    # No one imports this
    def test_simtools_analysis_baseanalyzers_DownloadAnalyzerTPI_import(self):
        from simtools.Analysis.BaseAnalyzers.DownloadAnalyzerTPI import DownloadAnalyzerTPI as test
        expected_methods = ['destroy', 'filter', 'finalize', 'from_cache', 'initialize', 'is_in_cache', 'keys',
                            'per_experiment', 'select_simulation_data', 'to_cache']
        self.report_mising_methods("DownloadAnalyzerTPI", expected_methods, test)

    # No one imports this
    def test_simtools_analysis_baseanalyzers_InsetChartAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers.InsetChartAnalyzer import InsetChartAnalyzer as test
        expected_methods = ['destroy', 'filter', 'finalize', 'initialize', 'per_experiment', 'select_simulation_data']
        self.report_mising_methods("InsetChartAnalyzer", expected_methods, test)

    # No one imports this
    def test_simtools_analysis_baseanalyzers_SimulationDirectoryMapAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers.SimulationDirectoryMapAnalyzer import SimulationDirectoryMapAnalyzer as test
        expected_methods = ['destroy', 'filter', 'finalize', 'initialize', 'per_experiment', 'select_simulation_data']
        self.report_mising_methods("SimulationDirectoryMapAnalyzer", expected_methods, test)

    @unittest.skip("Not a real test")
    def test_simtools_analysis_baseanalyzers_XXX_import(self):
        from simtools.Analysis.BaseAnalyzers import XXX as test
        expected_methods = []
        self.report_mising_methods("XXX", expected_methods, test)

    def test_simtools_analysis_ssmtnalysis_SSMTAnalysis_import(self):
        from simtools.Analysis.SSMTAnalysis import SSMTAnalysis as test
        expected_methods = ['analyze', 'validate_args']
        self.report_mising_methods("SSMTAnalysis", expected_methods, test)

    # from simtools.AssetManager.FileList
    def test_simtools_analysis_ssmtnalysis_FileList_import(self):
        from simtools.Analysis.SSMTAnalysis import FileList as test
        expected_methods = ['add_asset_file', 'add_file', 'add_path']
        self.report_mising_methods("FileList", expected_methods, test)

    # simtools.Managers.WorkItemManager
    def test_simtools_analysis_ssmtnalysis_WorkItemManager_import(self):
        from simtools.Analysis.SSMTAnalysis import WorkItemManager as test
        expected_methods = ['add_file', 'add_wo_arg', 'clear_user_files', 'clear_wo_args', 'create', 'execute', 'run',
                            'status', 'wait_for_finish']
        self.report_mising_methods("WorkItemManager", expected_methods, test)

    # random constants and stuff. Do we want these for every module?
    def test_simtools_analysis_imports(self):
        from simtools.Analysis.AnalyzeManager import EXCEPTION_KEY
        from simtools.Analysis.AnalyzeManager import WAIT_TIME
        from simtools.Analysis.AnalyzeManager import ANALYZE_TIMEOUT
        self.assertIn("EXCEPTION", EXCEPTION_KEY)
        self.assertIsNotNone(WAIT_TIME)
        self.assertIsNotNone(ANALYZE_TIMEOUT)
#</editor-fold>

    def test_simtools_DBLogging_spec(self):
        module = "simtools.DBLogging"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)
#<editor-fold desc='simtools.DBLogging space'>

    def test_simtools_dblogging_Schema_import(self):
        from simtools.DBLogging import Schema as test
        expected_methods = ['Base_logs','Timing']
        self.report_mising_methods("XXX", expected_methods, test)

    def test_simtools_dblogging_schema_LogRecord_import(self):
        from simtools.DBLogging.Schema import LogRecord as test
        expected_methods = ['created', 'cwd', 'exception', 'func_name', 'id', 'line_no', 'log_level', 'log_level_name',
                            'message', 'metadata', 'module', 'name', 'thread_name']
        self.report_mising_methods("LogRecord", expected_methods, test)

    def test_simtools_dblogging_schema_Timing_import(self):
        from simtools.DBLogging.Schema import Timing as test
        expected_methods = [ 'date', 'elapsed_time', 'extra_info', 'id', 'label', 'metadata', 'timing_id']
        self.report_mising_methods("Timing", expected_methods, test)

    def test_simtools_analysis_baseanalyzers_SQLiteHandler_import(self):
        from simtools.DBLogging.SQLiteHandler import SQLiteHandler as test
        expected_methods = ['acquire', 'addFilter', 'close', 'createLock', 'emit', 'filter', 'flush', 'format',
                            'formatDBTime', 'get_name', 'handle', 'handleError', 'name', 'release', 'removeFilter',
                            'setFormatter', 'setLevel', 'set_name']
        self.report_mising_methods("SQLiteHandler", expected_methods, test)
#</editor-fold>

    def test_simtools_DataAccess_spec(self):
        module = "simtools.DataAccess"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)
#<editor-fold desc='simtools.DataAccess space'>

    def test_simtools_dataaccess_BatchDataStore_import(self):
        import simtools.DataAccess.BatchDataStore as test
        expected_methods = ['Batch', 'BatchDataStore', 'BatchExperiment', 'BatchSimulation',
                            'joinedload', 'session_scope']
        self.report_mising_methods("BatchDataStore", expected_methods, test)

    @unittest.skip("Not a real test")
    def test_simtools_dataaccess_batchdatastore_XXX_import(self):
        from simtools.DataAccess.BatchDataStore import XXX as test
        expected_methods = []
        self.report_mising_methods("XXX", expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_Batch_import(self):
        from simtools.DataAccess.BatchDataStore import Batch as test
        expected_methods = ['date_created', 'experiments', 'get_experiment_ids', 'get_simulation_ids',
                            'id', 'metadata', 'name', 'simulations']
        self.report_mising_methods("Batch", expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_BatchDataStore_import(self):
        from simtools.DataAccess.BatchDataStore import BatchDataStore as test
        expected_methods = ['clear_batch', 'delete_batch', 'get_batch_by_id', 'get_batch_by_name',
                            'get_batch_list', 'remove_empty_batch', 'save_batch']
        self.report_mising_methods("BatchDataStore", expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_BatchExperiment_import(self):
        from simtools.DataAccess.BatchDataStore import BatchExperiment as test
        expected_methods = ['batch_id', 'exp_id', 'metadata']
        self.report_mising_methods("BatchExperiment", expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_BatchSimulation_import(self):
        from simtools.DataAccess.BatchDataStore import BatchSimulation as test
        expected_methods = ['batch_id', 'metadata', 'sim_id']
        self.report_mising_methods("BatchSimulation", expected_methods, test)

#</editor-fold>

    def test_simtools_Utilities_spec(self):
        module = "simtools.Utilities"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)
#<editor-fold desc='simtools.Utilities space'>


    def test_simtools_Utilities_import(self):
        import simtools.Utilities as test
        expected_methods = ['BetterConfigParser', 'COMPSCache', 'COMPSUtilities',
                            'CacheEnabled', 'Display', 'Distro', 'Encoding', 'Experiments',
                            'General', 'Initialization', 'Iterable', 'LocalOS', 'RetryDecorator',
                            'on_off', 'pluralize', 'verbose_timedelta']
        self.report_mising_methods("XXX", expected_methods, test)

    def test_simtools_utilities_betterconfigparser_BetterConfigParser_import(self):
        from simtools.Utilities.BetterConfigParser import BetterConfigParser as test
        expected_methods = ['add_section', 'clear', 'converters', 'defaults', 'get', 'getboolean',
                            'getfloat', 'getint', 'has_option', 'has_section', 'items', 'keys', 'options',
                            'optionxform', 'pop', 'popitem', 'read', 'read_dict', 'read_file', 'read_string',
                            'readfp', 'remove_option', 'remove_section', 'sections', 'set', 'setdefault',
                            'update', 'values', 'write']
        self.report_mising_methods("BetterConfigParser", expected_methods, test)

    def test_simtools_utilities_cacheenabled_CacheEnabled_import(self):
        from simtools.Utilities.CacheEnabled import CacheEnabled as test
        expected_methods = ['destroy_cache', 'initialize_cache']
        self.report_mising_methods("CacheEnabled", expected_methods, test)

    def test_simtools_utilities_compscache_COMPSCache_import(self):
        from simtools.Utilities.COMPSCache import COMPSCache as test
        expected_methods = ['add_experiment_to_cache', 'add_simulation_to_cache', 'collection', 'experiment',
                            'get_experiment_simulations', 'load_collection', 'load_experiment', 'load_simulation',
                            'query_collection', 'query_experiment', 'query_simulation', 'simulation']
        self.report_mising_methods("COMPSCache", expected_methods, test)

    def test_simtools_utilities_COMPSUtilities_import(self):
        from simtools.Utilities import COMPSUtilities as test
        expected_methods = ['AssetCollection', 'COMPS_login', 'Client', 'Experiment', 'QueryCriteria',
                            'SetupParser', 'Simulation', 'SimulationState', 'Suite', 'create_suite',
                            'datetime', 'delete_suite', 'download_asset_collection', 'experiment_is_running',
                            'experiment_needs_commission', 'exps_for_suite_id', 'get_all_experiments_for_user',
                            'get_asset_collection', 'get_asset_collection_by_id', 'get_asset_collection_by_tag',
                            'get_asset_collection_id_for_simulation_id', 'get_asset_files_for_simulation_id',
                            'get_experiment_by_id', 'get_experiment_ids_for_user', 'get_experiments_by_name',
                            'get_experiments_per_user_and_date', 'get_md5', 'get_semaphore',
                            'get_simulation_by_id', 'get_simulations_from_big_experiments', 'init_logging',
                            'is_comps_alive', 'logger', 'os', 'path_translations',
                            'pretty_display_assets_from_collection', 're', 'retry_function', 'shutil',
                            'sims_from_experiment', 'sims_from_experiment_id', 'sims_from_suite_id',
                            'stage_file', 'timedelta', 'translate_COMPS_path', 'workdirs_from_experiment_id',
                            'workdirs_from_simulations', 'workdirs_from_suite_id', 'zipfile']
        self.report_mising_methods("COMPSUtilities", expected_methods, test)

    def test_simtools_utilities_configobj_ConfigObj_import(self):
        from simtools.Utilities.ConfigObj import ConfigObj as test
        expected_methods = ['as_bool', 'as_float', 'as_int', 'as_list', 'clear', 'copy', 'dict', 'fromkeys',
                            'get', 'items', 'iteritems', 'iterkeys', 'itervalues', 'keys', 'merge', 'pop',
                            'popitem', 'reload', 'rename', 'reset', 'restore_default', 'restore_defaults',
                            'setdefault', 'update', 'validate', 'values', 'walk', 'write']
        self.report_mising_methods("ConfigObj", expected_methods, test)

    def test_simtools_utilities_DataFrame_import(self):
        from simtools.Utilities import DataFrame as test
        expected_methods = ['AgeBin', 'NotUpsampleable', 'upsample_agebin']
        self.report_mising_methods("DataFrame", expected_methods, test)

    def test_simtools_utilities_DiskSpaceUsage_import(self):
        from simtools.Utilities import DiskSpaceUsage as test
        expected_methods = ['DiskEncoder', 'DiskSpaceUsage', 'ExperimentInfo']
        self.report_mising_methods("DiskSpaceUsage", expected_methods, test)

    # doesn't appear to be used
    def test_simtools_utilities_Display_import(self):
        from simtools.Utilities import Display as test
        expected_methods = ['on_off', 'pluralize', 'verbose_timedelta']
        self.report_mising_methods("Display", expected_methods, test)

    # this appears to be all straight out of a 'distro' package. Should we just add to requirements?
    def test_simtools_utilities_Distro_import(self):
        from simtools.Utilities import Distro as test
        expected_methods = ['build_number', 'cached_property', 'codename', 'distro_release_attr',
                            'distro_release_info', 'id', 'info', 'like', 'linux_distribution',
                            'lsb_release_attr', 'lsb_release_info', 'main', 'major_version', 'minor_version',
                            'name', 'os_release_attr', 'os_release_info', 'test', 'uname_attr', 'uname_info',
                            'version', 'version_parts']
        self.report_mising_methods("Distro", expected_methods, test)

    def test_simtools_utilities_Encoding_import(self):
        from simtools.Utilities import Encoding as test
        expected_methods = ['GeneralEncoder', 'NumpyEncoder', 'cast_number', 'json_numpy_obj_hook']
        self.report_mising_methods("Encoding", expected_methods, test)

    def test_simtools_utilities_Experiments_import(self):
        from simtools.Utilities import Experiments as test
        expected_methods = ['COMPS_experiment_to_local_db', 'retrieve_experiment', 'retrieve_object',
                            'retrieve_simulation', 'validate_exp_name']
        self.report_mising_methods("Experiments", expected_methods, test)

    def test_simtools_utilities_General_import(self):
        from simtools.Utilities import General as test
        expected_methods = ['CommandlineGenerator', 'batch', 'batch_list', 'caller_name',
                            'copy_and_reset_StringIO', 'file_size', 'files_in_dir', 'get_md5',
                            'get_tools_revision', 'import_submodules', 'init_logging', 'is_running',
                            'labels', 'logger', 'logging_initialized', 'nostdout', 'remove_null_values',
                            'remove_special_chars', 'retrieve_item', 'retry_function', 'rmtree_f',
                            'rmtree_f_on_error', 'timestamp', 'timestamp_filename', 'utc_to_local', 'verbose']
        self.report_mising_methods("General", expected_methods, test)

    def test_simtools_utilities_Initialization_import(self):
        from simtools.Utilities import Initialization as test
        expected_methods = []
        self.report_mising_methods("Initialization", expected_methods, test)

    def test_simtools_utilities_Iterable_import(self):
        from simtools.Utilities import Iterable as test
        expected_methods = []
        self.report_mising_methods("Iterable", expected_methods, test)

    def test_simtools_utilities_LocalOS_import(self):
        from simtools.Utilities import LocalOS as test
        expected_methods = []
        self.report_mising_methods("LocalOS", expected_methods, test)

    def test_simtools_utilities_RetryDecorator_import(self):
        from simtools.Utilities import RetryDecorator as test
        expected_methods = []
        self.report_mising_methods("RetryDecorator", expected_methods, test)

    @unittest.skip("Not a real test")
    def test_simtools_utilities_XXX_import(self):
        from simtools.Utilities import XXX as test
        expected_methods = []
        self.report_mising_methods("XXX", expected_methods, test)

#</editor-fold>

    def test_simtools_ExperimentManager_spec(self):
        module = "simtools.ExperimentManager"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

    def test_simtools_SetupParser_spec(self):
        module = "simtools.SetupParser"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

# import each top level library from simtools

# import

    def report_mising_methods(self, namespace_name, methodlist, namespace, package_name=None):
        self.assertIsNotNone(namespace)
        listing = dir(namespace)
        expected_methods = methodlist
        messages = []
        for em in expected_methods:
            if em not in listing:
                messages.append(f"Property {em} not found in {namespace_name} namespace")
        self.assertEqual(0, len(messages), msg=f"Expected no errors, got {str(messages)}")
