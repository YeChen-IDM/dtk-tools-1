import unittest
import importlib

class TestInstallation(unittest.TestCase):

    def setUp(self):
        self.expected_methods = None
        pass

    def tearDown(self):
        self.expected_methods = None
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
        self.expected_methods = ["load_bin_file","load_csv_file","load_json_file",
                            "load_raw_file","load_txt_file","load_xlsx_file","parse"]
        from simtools.Analysis.OutputParser import SimulationOutputParser as test
        self.report_missing_methods("SimulationOutputParser", self.expected_methods, test)

    def test_simtools_analysis_DataRetrievalProcess_import(self):
        self.expected_methods = ['get_asset_files_for_simulation_id', 'itertools', 'os', 'retrieve_COMPS_AM_files',
                            'retrieve_SSMT_files', 'retrieve_data', 'retrieve_data_for_simulation', 'retry',
                            'set_exception', 'traceback']
        from simtools.Analysis import DataRetrievalProcess as test
        self.report_missing_methods("DataRetrievalProcess", self.expected_methods, test)

    def test_simtools_analysis_AnalyzeManager_import(self):
        self.expected_methods = ['animation', 'collections', 'init_logging', 'logger', 'on_off', 'os', 'pluralize',
                            'pool_worker_initializer', 're', 'retrieve_data', 'retrieve_experiment',
                            'retrieve_simulation', 'sys', 'time', 'verbose_timedelta']
        from simtools.Analysis import AnalyzeManager as test
        self.report_missing_methods("AnalyzeManager", self.expected_methods, test)

    def test_simtools_analysis_AnalyzeHelper_import(self):
        self.expected_methods = ['analyze', 'check_existing_batch', 'check_status', 'clean_batch', 'clear_batch',
                            'collect_analyzers', 'collect_experiments_simulations', 'collect_simulations',
                            'compare_two_ids_list', 'consolidate_experiments_with_options', 'create_batch',
                            'delete_batch', 'init_logging', 'list_batch', 'load_config_module', 'logger', 'os',
                            'retrieve_experiment', 'retrieve_item', 'retrieve_simulation', 'save_batch']
        from simtools.Analysis import AnalyzeHelper as test
        self.report_missing_methods("AnalyzeHelper", self.expected_methods, test)

    def test_simtools_analysis_baseanalyzers_BaseAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import BaseAnalyzer as test
        self.expected_methods = ['destroy', 'filter', 'finalize', 'initialize', 'per_experiment', 'select_simulation_data']
        self.report_missing_methods("BaseAnalyzer", self.expected_methods, test)
        pass

    def test_simtools_analysis_baseanalyzers_BaseCacheAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import BaseCacheAnalyzer as test
        self.expected_methods = ['destroy', 'filter', 'finalize', 'from_cache', 'initialize', 'is_in_cache', 'keys',
                            'per_experiment', 'select_simulation_data', 'to_cache']
        self.report_missing_methods("BaseAnalyzer", self.expected_methods, test)
        pass

    def test_simtools_analysis_baseanalyzers_BaseCalibrationAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import BaseCalibrationAnalyzer as test
        self.expected_methods = ['cache', 'destroy', 'filter', 'finalize', 'initialize', 'per_experiment',
                            'select_simulation_data']
        self.report_missing_methods("BaseAnalyzer", self.expected_methods, test)
        pass

    def test_simtools_analysis_baseanalyzers_DownloadAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers import DownloadAnalyzer as test
        self.expected_methods = ['destroy', 'filter', 'finalize', 'get_sim_folder', 'initialize', 'per_experiment',
                            'select_simulation_data']
        self.report_missing_methods("BaseAnalyzer", self.expected_methods, test)
        pass

    # No one imports this
    def test_simtools_analysis_baseanalyzers_DownloadAnalyzerTPI_import(self):
        from simtools.Analysis.BaseAnalyzers.DownloadAnalyzerTPI import DownloadAnalyzerTPI as test
        self.expected_methods = ['destroy', 'filter', 'finalize', 'from_cache', 'initialize', 'is_in_cache', 'keys',
                            'per_experiment', 'select_simulation_data', 'to_cache']
        self.report_missing_methods("DownloadAnalyzerTPI", self.expected_methods, test)

    # No one imports this
    def test_simtools_analysis_baseanalyzers_InsetChartAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers.InsetChartAnalyzer import InsetChartAnalyzer as test
        self.expected_methods = ['destroy', 'filter', 'finalize', 'initialize', 'per_experiment', 'select_simulation_data']
        self.report_missing_methods("InsetChartAnalyzer", self.expected_methods, test)

    # No one imports this
    def test_simtools_analysis_baseanalyzers_SimulationDirectoryMapAnalyzer_import(self):
        from simtools.Analysis.BaseAnalyzers.SimulationDirectoryMapAnalyzer import SimulationDirectoryMapAnalyzer as test
        self.expected_methods = ['destroy', 'filter', 'finalize', 'initialize', 'per_experiment', 'select_simulation_data']
        self.report_missing_methods("SimulationDirectoryMapAnalyzer", self.expected_methods, test)

    def test_simtools_analysis_ssmtnalysis_SSMTAnalysis_import(self):
        from simtools.Analysis.SSMTAnalysis import SSMTAnalysis as test
        self.expected_methods = ['analyze', 'validate_args']
        self.report_missing_methods("SSMTAnalysis", self.expected_methods, test)

    # from simtools.AssetManager.FileList
    def test_simtools_analysis_ssmtnalysis_FileList_import(self):
        from simtools.Analysis.SSMTAnalysis import FileList as test
        self.expected_methods = ['add_asset_file', 'add_file', 'add_path']
        self.report_missing_methods("FileList", self.expected_methods, test)

    # simtools.Managers.WorkItemManager
    def test_simtools_analysis_ssmtnalysis_WorkItemManager_import(self):
        from simtools.Analysis.SSMTAnalysis import WorkItemManager as test
        self.expected_methods = ['add_file', 'add_wo_arg', 'clear_user_files', 'clear_wo_args', 'create', 'execute', 'run',
                            'status', 'wait_for_finish']
        self.report_missing_methods("WorkItemManager", self.expected_methods, test)

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
        self.expected_methods = ['Base_logs','Timing']
        self.report_missing_methods("XXX", self.expected_methods, test)

    def test_simtools_dblogging_schema_LogRecord_import(self):
        from simtools.DBLogging.Schema import LogRecord as test
        self.expected_methods = ['created', 'cwd', 'exception', 'func_name', 'id', 'line_no', 'log_level', 'log_level_name',
                            'message', 'metadata', 'module', 'name', 'thread_name']
        self.report_missing_methods("LogRecord", self.expected_methods, test)

    def test_simtools_dblogging_schema_Timing_import(self):
        from simtools.DBLogging.Schema import Timing as test
        self.expected_methods = [ 'date', 'elapsed_time', 'extra_info', 'id', 'label', 'metadata', 'timing_id']
        self.report_missing_methods("Timing", self.expected_methods, test)

    def test_simtools_analysis_baseanalyzers_SQLiteHandler_import(self):
        from simtools.DBLogging.SQLiteHandler import SQLiteHandler as test
        self.expected_methods = ['acquire', 'addFilter', 'close', 'createLock', 'emit', 'filter', 'flush', 'format',
                            'formatDBTime', 'get_name', 'handle', 'handleError', 'name', 'release', 'removeFilter',
                            'setFormatter', 'setLevel', 'set_name']
        self.report_missing_methods("SQLiteHandler", self.expected_methods, test)
#</editor-fold>

    def test_simtools_DataAccess_spec(self):
        module = "simtools.DataAccess"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.DataAccess space'>

    def test_simtools_dataaccess_BatchDataStore_import(self):
        import simtools.DataAccess.BatchDataStore as test
        self.expected_methods = ['Batch', 'BatchDataStore', 'BatchExperiment', 'BatchSimulation',
                            'joinedload', 'session_scope']
        self.report_missing_methods("BatchDataStore", self.expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_Batch_import(self):
        from simtools.DataAccess.BatchDataStore import Batch as test
        self.expected_methods = ['date_created', 'experiments', 'get_experiment_ids', 'get_simulation_ids',
                            'id', 'metadata', 'name', 'simulations']
        self.report_missing_methods("Batch", self.expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_BatchDataStore_import(self):
        from simtools.DataAccess.BatchDataStore import BatchDataStore as test
        self.expected_methods = ['clear_batch', 'delete_batch', 'get_batch_by_id', 'get_batch_by_name',
                            'get_batch_list', 'remove_empty_batch', 'save_batch']
        self.report_missing_methods("BatchDataStore", self.expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_BatchExperiment_import(self):
        from simtools.DataAccess.BatchDataStore import BatchExperiment as test
        self.expected_methods = ['batch_id', 'exp_id', 'metadata']
        self.report_missing_methods("BatchExperiment", self.expected_methods, test)

    def test_simtools_dataaccess_batchdatastore_BatchSimulation_import(self):
        from simtools.DataAccess.BatchDataStore import BatchSimulation as test
        self.expected_methods = ['batch_id', 'metadata', 'sim_id']
        self.report_missing_methods("BatchSimulation", self.expected_methods, test)
#</editor-fold>

    def test_simtools_Utilities_spec(self):
        module = "simtools.Utilities"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)
#<editor-fold desc='simtools.Utilities space'>


    def test_simtools_Utilities_import(self):
        import simtools.Utilities as test
        self.expected_methods = ['Display', 'Iterable', 'pluralize', 'verbose_timedelta']
        self.report_missing_methods("simtools.Utilities", self.expected_methods, test)

    def test_simtools_utilities_betterconfigparser_BetterConfigParser_import(self):
        from simtools.Utilities.BetterConfigParser import BetterConfigParser as test
        self.expected_methods = ['add_section', 'clear', 'converters', 'defaults', 'get', 'getboolean',
                            'getfloat', 'getint', 'has_option', 'has_section', 'items', 'keys', 'options',
                            'optionxform', 'pop', 'popitem', 'read', 'read_dict', 'read_file', 'read_string',
                            'readfp', 'remove_option', 'remove_section', 'sections', 'set', 'setdefault',
                            'update', 'values', 'write']
        self.report_missing_methods("BetterConfigParser", self.expected_methods, test)

    def test_simtools_utilities_cacheenabled_CacheEnabled_import(self):
        from simtools.Utilities.CacheEnabled import CacheEnabled as test
        self.expected_methods = ['destroy_cache', 'initialize_cache']
        self.report_missing_methods("CacheEnabled", self.expected_methods, test)

    def test_simtools_utilities_compscache_COMPSCache_import(self):
        from simtools.Utilities.COMPSCache import COMPSCache as test
        self.expected_methods = ['add_experiment_to_cache', 'add_simulation_to_cache', 'collection', 'experiment',
                            'get_experiment_simulations', 'load_collection', 'load_experiment', 'load_simulation',
                            'query_collection', 'query_experiment', 'query_simulation', 'simulation']
        self.report_missing_methods("COMPSCache", self.expected_methods, test)

    def test_simtools_utilities_COMPSUtilities_import(self):
        from simtools.Utilities import COMPSUtilities as test
        self.expected_methods = ['AssetCollection', 'COMPS_login', 'Client', 'Experiment', 'QueryCriteria',
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
        self.report_missing_methods("COMPSUtilities", self.expected_methods, test)

    def test_simtools_utilities_configobj_ConfigObj_import(self):
        from simtools.Utilities.ConfigObj import ConfigObj as test
        self.expected_methods = ['as_bool', 'as_float', 'as_int', 'as_list', 'clear', 'copy', 'dict', 'fromkeys',
                            'get', 'items', 'iteritems', 'iterkeys', 'itervalues', 'keys', 'merge', 'pop',
                            'popitem', 'reload', 'rename', 'reset', 'restore_default', 'restore_defaults',
                            'setdefault', 'update', 'validate', 'values', 'walk', 'write']
        self.report_missing_methods("ConfigObj", self.expected_methods, test)

    def test_simtools_utilities_DataFrame_import(self):
        from simtools.Utilities import DataFrame as test
        self.expected_methods = ['AgeBin', 'NotUpsampleable', 'upsample_agebin']
        self.report_missing_methods("DataFrame", self.expected_methods, test)

    def test_simtools_utilities_DiskSpaceUsage_import(self):
        from simtools.Utilities import DiskSpaceUsage as test
        self.expected_methods = ['DiskEncoder', 'DiskSpaceUsage', 'ExperimentInfo']
        self.report_missing_methods("DiskSpaceUsage", self.expected_methods, test)

    # doesn't appear to be used
    def test_simtools_utilities_Display_import(self):
        from simtools.Utilities import Display as test
        self.expected_methods = ['on_off', 'pluralize', 'verbose_timedelta']
        self.report_missing_methods("Display", self.expected_methods, test)

    # this appears to be all straight out of a 'distro' package. Should we just add to requirements?
    def test_simtools_utilities_Distro_import(self):
        from simtools.Utilities import Distro as test
        self.expected_methods = ['build_number', 'cached_property', 'codename', 'distro_release_attr',
                            'distro_release_info', 'id', 'info', 'like', 'linux_distribution',
                            'lsb_release_attr', 'lsb_release_info', 'main', 'major_version', 'minor_version',
                            'name', 'os_release_attr', 'os_release_info', 'test', 'uname_attr', 'uname_info',
                            'version', 'version_parts']
        self.report_missing_methods("Distro", self.expected_methods, test)

    def test_simtools_utilities_Encoding_import(self):
        from simtools.Utilities import Encoding as test
        self.expected_methods = ['GeneralEncoder', 'NumpyEncoder', 'cast_number', 'json_numpy_obj_hook']
        self.report_missing_methods("Encoding", self.expected_methods, test)

    def test_simtools_utilities_Experiments_import(self):
        from simtools.Utilities import Experiments as test
        self.expected_methods = ['COMPS_experiment_to_local_db', 'retrieve_experiment', 'retrieve_object',
                            'retrieve_simulation', 'validate_exp_name']
        self.report_missing_methods("Experiments", self.expected_methods, test)

    def test_simtools_utilities_General_import(self):
        from simtools.Utilities import General as test
        self.expected_methods = ['CommandlineGenerator', 'batch', 'batch_list', 'caller_name',
                            'copy_and_reset_StringIO', 'file_size', 'files_in_dir', 'get_md5',
                            'get_tools_revision', 'import_submodules', 'init_logging', 'is_running',
                            'labels', 'logger', 'logging_initialized', 'nostdout', 'remove_null_values',
                            'remove_special_chars', 'retrieve_item', 'retry_function', 'rmtree_f',
                            'rmtree_f_on_error', 'timestamp', 'timestamp_filename', 'utc_to_local', 'verbose']
        self.report_missing_methods("General", self.expected_methods, test)

    def test_simtools_utilities_Initialization_import(self):
        from simtools.Utilities import Initialization as test
        self.expected_methods = ['grab_HPC_overrides', 'initialize_SetupParser_from_args',
                            'load_config', 'load_config_module']
        self.report_missing_methods("Initialization", self.expected_methods, test)

    def test_simtools_utilities_Matlab_import(self):
        from simtools.Utilities import Matlab as test
        self.expected_methods = ['read_mat_points_file']
        self.report_missing_methods("Matlab", self.expected_methods, test)

    def test_simtools_utilities_LocalOS_import(self):
        from simtools.Utilities import LocalOS as test
        self.expected_methods = ['get_linux_distribution', 'LocalOS']
        self.report_missing_methods("LocalOS", self.expected_methods, test)

    def test_simtools_utilities_RetryDecorator_import(self):
        from simtools.Utilities import RetryDecorator as test
        self.expected_methods = ['retry']
        self.report_missing_methods("RetryDecorator", self.expected_methods, test)

    def test_simtools_utilities_simulationdirectorymap_SimulationDirectoryMap_import(self):
        from simtools.Utilities.SimulationDirectoryMap import SimulationDirectoryMap as test
        self.expected_methods = ['dir_map', 'get_simulation_path', 'preload_experiment', 'single_simulation_dir']
        self.report_missing_methods("SimulationDirectoryMap", self.expected_methods, test)

    def test_simtools_utilities_Tabulate_import(self):
        from simtools.Utilities import Tabulate as test
        self.expected_methods = ['Line', 'MIN_PADDING', 'PRESERVE_WHITESPACE', 'TableFormat', 'WIDE_CHARS_MODE',
                            'basestring', 'izip_longest', 'multiline_formats', 'simple_separated_format',
                            'tabulate', 'tabulate_formats']
        self.report_missing_methods("Tabulate", self.expected_methods, test)

#</editor-fold>

    def test_simtools_ExperimentManager_spec(self):
        module = "simtools.ExperimentManager"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.ExperimentManager space'>

    def test_simtools_exeperimentmanager_BaseExperimentManager_import(self):
        from simtools.ExperimentManager import BaseExperimentManager as test
        self.expected_methods = ['BaseExperimentManager', 'simulations_expected', 'logger', 'current_dir']
        self.report_missing_methods("BaseExperimentManager", self.expected_methods, test)

    def test_simtools_exeperimentmanager_baseexperimentmanager_BaseExperimentManager_import(self):
        from simtools.ExperimentManager.BaseExperimentManager import BaseExperimentManager as test
        self.expected_methods = ['any_failed_or_cancelled', 'cancel_experiment', 'cancel_simulations', 'check_overseer',
                            'clean_experiment_name', 'commission_simulations', 'create_experiment',
                            'create_simulations', 'create_suite', 'destroy_cache', 'done_commissioning', 'finished',
                            'get_simulation_creator', 'get_simulation_status', 'hard_delete', 'initialize_cache',
                            'kill', 'kill_simulation', 'location', 'print_status', 'refresh_experiment',
                            'run_simulations', 'status_failed', 'status_finished', 'status_succeeded', 'succeeded',
                            'validate_input_files', 'wait_for_finished']
        self.report_missing_methods("BaseExperimentManager", self.expected_methods, test)

    def test_simtools_exeperimentmanager_CompsExperimentManager_import(self):
        from simtools.ExperimentManager import CompsExperimentManager as test
        self.expected_methods = ['CompsExperimentManager', 'logger']
        self.report_missing_methods("CompsExperimentManager", self.expected_methods, test)

    def test_simtools_exeperimentmanager_compsexperimentmanager_CompsExperimentManager_import(self):
        from simtools.ExperimentManager.CompsExperimentManager import CompsExperimentManager as test
        self.expected_methods = ['MAX_SUBDIRECTORY_LENGTH', 'any_failed_or_cancelled', 'cancel_experiment',
                            'cancel_simulations', 'check_overseer', 'clean_experiment_name', 'commission_simulations',
                            'create_experiment', 'create_simulation', 'create_simulations', 'create_suite',
                            'destroy_cache', 'done_commissioning', 'finished', 'get_simulation_creator',
                            'get_simulation_status', 'hard_delete', 'initialize_cache', 'kill', 'kill_simulation',
                            'location', 'merge_tags', 'print_status', 'refresh_experiment', 'run_simulations',
                            'status_failed', 'status_finished', 'status_succeeded', 'succeeded',
                            'validate_input_files', 'wait_for_finished']
        self.report_missing_methods("CompsExperimentManager", self.expected_methods, test)

    def test_simtools_exeperimentmanager_ExperimentManagerFactory_import(self):
        from simtools.ExperimentManager import ExperimentManagerFactory as test
        self.expected_methods = ['ExperimentManagerFactory', 'logger']
        self.report_missing_methods("ExperimentManagerFactory", self.expected_methods, test)

    def test_simtools_exeperimentmanager_experimentmanagerfactory_ExperimentManagerFactory_import(self):
        from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory as test
        self.expected_methods = ['from_cb', 'from_experiment', 'from_model', 'from_setup', 'init']
        self.report_missing_methods("ExperimentManagerFactory", self.expected_methods, test)

    def test_simtools_exeperimentmanager_LocalExperimentManager_import(self):
        from simtools.ExperimentManager import LocalExperimentManager as test
        self.expected_methods = ['LocalExperimentManager', 'logger']
        self.report_missing_methods("LocalExperimentManager", self.expected_methods, test)

    def test_simtools_exeperimentmanager_localexperimentmanager_LocalExperimentManager_import(self):
        from simtools.ExperimentManager.LocalExperimentManager import LocalExperimentManager as test
        self.expected_methods = ['any_failed_or_cancelled', 'cancel_experiment', 'cancel_simulations', 'check_overseer',
                            'clean_experiment_name', 'commission_simulations', 'create_experiment',
                            'create_simulations', 'create_suite', 'destroy_cache', 'done_commissioning',
                            'experiment', 'finished', 'get_simulation_creator', 'get_simulation_status', 'hard_delete',
                            'initialize_cache', 'kill', 'kill_simulation', 'location', 'needs_commissioning',
                            'print_status', 'refresh_experiment', 'run_simulations', 'status_failed',
                            'status_finished', 'status_succeeded', 'succeeded', 'validate_input_files',
                            'wait_for_finished']
        self.report_missing_methods("LocalExperimentManager", self.expected_methods, test)

#</editor-fold>
    def test_simtools_SetupParser_spec(self):
        module = "simtools.SetupParser"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.SetupParser space'>

    def test_simtools_SetupParser_import(self):
        import simtools.SetupParser as test
        self.expected_methods = ['BetterConfigParser', 'SetupParser', 'SetupParserMeta',
                            'current_dir', 'fasteners', 'init_logging', 'logger']
        self.report_missing_methods("simtools.SetupParser", self.expected_methods, test)

    def test_simtools_setupparser_SetupParser_import(self):
        from simtools.SetupParser import SetupParser as test
        self.expected_methods = ['AlreadyInitialized', 'InvalidBlock', 'MissingIniBlock', 'MissingIniFile',
                            'NotInitialized', 'TemporaryBlock', 'TemporarySetup', 'default_block',
                            'default_file', 'get', 'getboolean', 'has_option', 'ini_filename', 'items',
                            'load_schema', 'old_style_instantiation', 'override_block',
                            'resolve_type_inheritance', 'set', 'validate']
        self.report_missing_methods("SetupParser", self.expected_methods, test)

    def test_simtools_setupparser_SetupParserMeta_import(self):
        from simtools.SetupParser import SetupParserMeta as test
        self.expected_methods = ['_uninit', 'init', 'initialized', 'singleton']
        self.report_missing_methods("SetupParserMeta", self.expected_methods, test)

#</editor-fold>

    def test_simtools_ModBuilder_spec(self):
        module = "simtools.ModBuilder"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.ModBuilder space'>

    def test_simtools_ModBuilder_import(self):
        import simtools.ModBuilder as test
        self.expected_methods = ['ModBuilder', 'ModFn', 'ModList', 'SingleSimulationBuilder']
        self.report_missing_methods("ModBuilder", self.expected_methods, test)

    def test_simtools_modbuilder_ModFn_import(self):
        from simtools.ModBuilder import ModFn as test
        self.expected_methods = ['__init__','__call__']
        self.report_missing_methods("ModFn", self.expected_methods, test)

    def test_simtools_modbuilder_ModBuilder_import(self):
        from simtools.ModBuilder import ModBuilder as test
        self.expected_methods = ['from_combos', 'from_list', 'metadata', 'set_mods']
        self.report_missing_methods("ModBuilder", self.expected_methods, test)

#</editor-fold>

    def test_simtools_Monitor_spec(self):
        module = "simtools.Monitor"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.Monitor space'>

    def test_simtools_monitor_SimulationMonitor_import(self):
        from simtools.Monitor import SimulationMonitor as test
        self.expected_methods = ['__init__', 'query']
        self.report_missing_methods("SimulationMonitor", self.expected_methods, test)

    def test_simtools_monitor_CompsSimulationMonitor_import(self):
        from simtools.Monitor import CompsSimulationMonitor as test
        self.expected_methods = ['__init__', 'query']
        self.report_missing_methods("CompsSimulationMonitor", self.expected_methods, test)

#</editor-fold>

    def test_simtools_SimulationCreator_spec(self):
        module = "simtools.SimulationCreator"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.SimulationCreator space'>

    def test_simtools_simulationcreator_BaseSimulationCreator_import(self):
        from simtools.SimulationCreator.BaseSimulationCreator import BaseSimulationCreator as test
        self.expected_methods = ['add_files_to_simulation', 'authkey', 'create_simulation', 'daemon', 'exitcode',
                            'ident', 'is_alive', 'join', 'name', 'pid', 'post_creation', 'pre_creation',
                            'process', 'process_batch', 'run', 'save_batch', 'sentinel', 'set_tags_to_simulation',
                            'start', 'terminate']
        self.report_missing_methods("BaseSimulationCreator", self.expected_methods, test)

    def test_simtools_simulationcreator_LocalSimulationCreator_import(self):
        from simtools.SimulationCreator import LocalSimulationCreator as test
        self.expected_methods = ['BaseSimulationCreator', 'LocalSim', 'LocalSimulationCreator']
        self.report_missing_methods("LocalSimulationCreator", self.expected_methods, test)

    def test_simtools_simulationcreator_localsimulationcreator_LocalSimulationCreator_import(self):
        from simtools.SimulationCreator.LocalSimulationCreator import LocalSimulationCreator as test
        self.expected_methods = ['add_files_to_simulation', 'authkey', 'create_simulation', 'daemon', 'exitcode',
                            'generate_UUID', 'ident', 'is_alive', 'join', 'name', 'pid', 'post_creation',
                            'pre_creation', 'process', 'process_batch', 'run', 'save_batch', 'sentinel',
                            'set_tags_to_simulation', 'start', 'terminate']
        self.report_missing_methods("LocalSimulationCreator", self.expected_methods, test)

    def test_simtools_simulationcreator_COMPSSimulationCreator_import(self):
        from simtools.SimulationCreator.COMPSSimulationCreator import COMPSSimulationCreator as test
        self.expected_methods = ['add_files_to_simulation', 'authkey', 'create_simulation', 'daemon', 'exitcode',
                            'ident', 'is_alive', 'join', 'name', 'pid', 'post_creation', 'pre_creation', 'process',
                            'process_batch', 'run', 'save_batch', 'sentinel', 'set_tags_to_simulation', 'start',
                            'terminate']
        self.report_missing_methods("COMPSSimulationCreator", self.expected_methods, test)

#</editor-fold>

    def test_simtools_SimulationRunner_spec(self):
        module = "simtools.SimulationRunner"
        spec = importlib.util.find_spec(module)
        self.assertEqual(spec.name, module)

#<editor-fold desc='simtools.SimulationRunner space'>

    def test_simtools_simulationrunner_LocalRunner_import(self):
        from simtools.SimulationRunner.LocalRunner import LocalSimulationRunner as test
        self.expected_methods = ['MONITOR_SLEEP', 'check_state', 'last_status_line', 'monitor', 'run', 'update_status']
        self.report_missing_methods("LocalSimulationRunner", self.expected_methods, test)

    def test_simtools_simulationrunner_basesimulationrunner_BaseSimulationRunner_import(self):
        from simtools.SimulationRunner.BaseSimulationRunner import BaseSimulationRunner as test
        self.expected_methods = ['MONITOR_SLEEP', 'monitor', 'run']
        self.report_missing_methods("BaseSimulationRunner", self.expected_methods, test)

    def test_simtools_simulationrunner_compsrunner_COMPSSimulationRunner_import(self):
        from simtools.SimulationRunner.COMPSRunner import COMPSSimulationRunner as test
        self.expected_methods = ['MONITOR_SLEEP', 'monitor', 'run']
        self.report_missing_methods("COMPSSimulationRunner", self.expected_methods, test)

#</editor-fold>

    def test_simtools_OutputParser_import(self):
        import simtools.OutputParser as test
        self.expected_methods = ['CompsDTKOutputParser', 'SimulationOutputParser']
        self.report_missing_methods("simtools.OutputParser", self.expected_methods, test)

#<editor-fold desc='simtools.OutputParser space">

    def test_simtools_outputparser_CompsDTKOutputParser_import(self):
        from simtools.OutputParser import CompsDTKOutputParser as test
        self.expected_methods = ['asset_service', 'createSimDirectoryMap', 'load_all_files', 'get_sim_dir']
        self.report_missing_methods("CompsDTKOutputParser", self.expected_methods, test)

    def test_simtools_outputparser_SimulationOutputParser_import(self):
        from simtools.OutputParser import SimulationOutputParser as test
        self.expected_methods = ['daemon', 'getName', 'get_last_megabyte', 'get_path', 'get_sim_dir', 'ident',
                                 'isAlive', 'isDaemon', 'is_alive', 'join', 'load_all_files', 'load_bin_file',
                                 'load_csv_file', 'load_json_file', 'load_raw_file', 'load_single_file',
                                 'load_txt_file', 'load_xlsx_file', 'name', 'run', 'setDaemon', 'setName',
                                 'sim_path', 'start']
        self.report_missing_methods("SimulationOutputParser", self.expected_methods, test)

#</editor-fold>

    def test_simtools_Overseer_import(self):
        import simtools.Overseer as test
        self.expected_methods = ['LogCleaner', 'logger']
        self.report_missing_methods("Overseer", self.expected_methods, test)

    def report_missing_methods(self, namespace_name, methodlist, namespace, package_name=None):
        self.assertIsNotNone(namespace)
        listing = dir(namespace)
        missing_properties = []
        for em in methodlist:
            if em not in listing:
                missing_properties.append(em)
        self.assertEqual(0, len(missing_properties), msg=f"Expected no missing properties in {namespace_name},"
                                                         f" missing: {str(missing_properties)}")
