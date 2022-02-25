import inspect
import logging
import configparser
import json
import os
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from hyperopt import hp
from matplotlib import pyplot as plt

import u_utils.u_utils as utils
import z_setting_parameters as settings


def create_parameters_log_file(dir_runtime_files, params):
    """
    Creates a log file with all given parameters for each iteration.

    :param dir_runtime_files: The folder of the current run.
    :param params: A grid of collections of parameters that is used to run all different combinations of the containing parameters.
    :return: None
    """

    log_file_content = \
        [['exogenous_path_data_sources', settings.path_data_sources, 'Path of folder with source files'],
         ['exogenous_dir_runtime_files', settings.dir_runtime_files,
          'Folder containing files read and written during runtime'],
         ['exogenous_dir_runtime_files_iteration', settings.dir_runtime_files_iteration,
          'Folder of one iteration containing files read and written during runtime'],
         ['exogenous_filename_room_separation', settings.filename_room_separation, 'Filename of room separation file.'],
         ['exogenous_filename_adjacency_matrix', settings.filename_adjacency_matrix, 'Filename of adjacency matrix.'],
         ['exogenous_filename_parameters_file', settings.filename_parameters_file, 'Filename of parameters file.'],
         ['exogenous_filename_adjacency_plot', settings.filename_adjacency_plot, 'Filename of adjacency plot.'],
         ['exogenous_filename_sensor_data', settings.filename_sensor_data, 'Filename of sensor data file.'],
         ['exogenous_rel_dir_name_sensor_data', settings.rel_dir_name_sensor_data,
          'The folder with sensor data (relative from data sources directory).'],
         ['exogenous_csv_delimiter_sensor_data', settings.csv_delimiter_sensor_data,
          'Delimiter of the columns in csv file of sensor data (input)'],
         ['exogenous_csv_header_sensor_data', settings.csv_header_sensor_data,
          'Indicator at which line the data starts.'],
         ['exogenous_csv_parse_dates_sensor_data', settings.csv_parse_dates_sensor_data,
          'Columns that should get parsed as a date.'],
         ['exogenous_csv_dtype_sensor_data', settings.csv_dtype_sensor_data,
          'An assignment of data types to columns in sensor data file.'],
         ['exogenous_filename_traces_raw', settings.filename_traces_raw, 'Filename of traces file.'],
         ['exogenous_csv_delimiter_traces', settings.csv_delimiter_traces, 'The char each column is divided by.'],
         ['exogenous_vectorization_type_list', settings.vectorization_type_list,
          'range for vectorization type (parameter optimization)'],
         ['exogenous_filename_log_export', settings.filename_log_export, 'Filename of log export file.'],
         ['exogenous_dir_petri_net_files', settings.dir_petri_net_files,
          'Directory in which petri net export files are saved.'],
         ['exogenous_filename_petri_net', settings.filename_petri_net, 'Filename of petri net pnml file.'],
         ['exogenous_filename_petri_net_image', settings.filename_petri_net_image, 'Filename of petri net image file.'],
         ['exogenous_dir_dfg_files', settings.dir_dfg_files, 'Directory in which dfg images are saved.'],
         ['exogenous_filename_dfg_cluster', settings.filename_dfg_cluster, 'Filename of dfg cluster image files.'],
         ['exogenous_filename_dfg', settings.filename_dfg, 'Filename of dfg image file.'],
         ['exogenous_rel_proportion_dfg_threshold', settings.rel_proportion_dfg_threshold,
          'Threshold for number of sensor activations at which a sensor is shown in dfg (relative to max occurrences of a sensor).'],
         ['filename_log_file', settings.filename_log_file + '.log', 'Filename of logfile.'],
         ['exogenous_logging_level', settings.logging_level, 'Lvl of logging for log file and console. (10=Debugging)'],
         ['exogenous_zero_distance_value', params['zero_distance_value'],
          'Number representing zero distance to other sensors. (used in creation of distance_matrix_real_world matrix)'],
         # ['exogenous_trace_length_limit', params['trace_length_limit'],
         #  'Maximum length of traces. (in case length mode is used to separate raw-traces)']
         ]

    get_trace = getattr(sys, 'gettrace', None)
    if get_trace() is None:
        log_file_content.append(['debug-mode', 'Deactivated', 'Debug Activated/Deactivated'])
    elif get_trace():
        log_file_content.append(['debug-mode', 'Activated', 'Debug Activated/Deactivated'])

    pandas_log_file_content = pd.DataFrame.from_records(log_file_content, columns=['Variable', 'Value', 'Description'])
    # sort data frame
    pandas_log_file_content = pandas_log_file_content.sort_values(by='Variable')
    target_folder = settings.path_data_sources + dir_runtime_files

    # Create a file with used parameters
    pandas_log_file_content.to_csv(path_or_buf=target_folder + settings.filename_parameters_file,
                                   index=False,
                                   sep=';')


def append_to_log_file(new_entry_to_log_variable,
                       new_entry_to_log_value,
                       path_data_sources,
                       filename_parameters_file,
                       dir_runtime_files,
                       new_entry_to_log_description=None):
    target_file = path_data_sources + dir_runtime_files + filename_parameters_file
    log_data = pd.read_csv(filepath_or_buffer=target_file,
                           sep=';')
    new_entry = pd.DataFrame([[new_entry_to_log_variable, new_entry_to_log_value, new_entry_to_log_description]],
                             columns=['Variable', 'Value', 'Description'])
    log_data = pd.concat(objs=[log_data, new_entry],
                         ignore_index=True)
    log_data = log_data.sort_values(by='Variable')
    log_data.to_csv(path_or_buf=target_file,
                    index=False,
                    sep=';')


def append_to_performance_documentation_file(path_data_sources,
                                             dir_runtime_files,
                                             filename_benchmark,
                                             csv_delimiter_benchmark,
                                             list_of_properties):
    # start timer
    t0_runtime = timeit.default_timer()
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])

    folder_name = dir_runtime_files.split('/')[-2]

    benchmark_file_path = Path(path_data_sources + dir_runtime_files.split('/')[0] + '/' + filename_benchmark)
    pass
    if benchmark_file_path.is_file():
        logger.debug("Reading preexisting benchmark file.")

        pd_benchmark_old = pd.read_csv(benchmark_file_path,
                                       sep=csv_delimiter_benchmark,
                                       header=0,
                                       index_col=0)
        pd_benchmark_add = pd.DataFrame(list_of_properties, index=[folder_name])
        pd_benchmark_new = pd.concat([pd_benchmark_old, pd_benchmark_add], sort=False)

        pd_benchmark_new.to_csv(benchmark_file_path, sep=csv_delimiter_benchmark)

    # if does not exist, create file
    else:
        logger.debug("Creating new benchmark file.")
        pd_benchmark = pd.DataFrame(list_of_properties, index=[folder_name])

        pd_benchmark.to_csv(benchmark_file_path, sep=';')
        # stop timer
        t1_runtime = timeit.default_timer()
        # calculate runtime
        runtime = np.round(t1_runtime - t0_runtime, 1)

        logger.debug("Saving entry to benchmark file took %s seconds.",
                     runtime)


def convert_search_space_into_selection(search_space_min,
                                        search_space_max,
                                        step_length):
    """
    Creates a selection of values which represents the given search place. The number of the values is defined by the
    step length and the size of the given search space.

    :param search_space_min:    Minimal limit of the search space.
    :param search_space_max:    Maximal limit of the search space.
    :param step_length:         The step length between each value the search space will divided in.

    :return: a list containing numeric values from the search space
    """
    curr_value = search_space_min
    selection = []

    while curr_value < search_space_max:
        selection.append(curr_value)
        curr_value += step_length

    selection.append(search_space_max)

    return selection


def check_settings(zero_distance_value_min, zero_distance_value_max,
                   # trace_length_limit_min,
                   # trace_length_limit_max,
                   miner_type,
                   miner_type_list, metric_to_be_maximised, metric_to_be_maximised_list):
    # checks settings for correctness (if they are invalid the execution get stopped)
    settings_valid = True

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    # checking different settings
    if zero_distance_value_min > zero_distance_value_max:
        logger.error("'zero_distance_value_min' has to be <= 'zero_distance_value_max'")
        settings_valid = False

    # if trace_length_limit_min > trace_length_limit_max:
    #     logger.error("'trace_length_limit_min' has to be <= 'trace_length_limit_max'")
    #     settings_valid = False

    if miner_type not in miner_type_list:
        error_msg = str("'" + miner_type +
                        "' is not a valid choice for a miner. Please choose one of the following: " +
                        str(miner_type_list))
        logger.error(error_msg)
        settings_valid = False

    if metric_to_be_maximised not in metric_to_be_maximised_list:
        error_msg = str("'" + metric_to_be_maximised +
                        "' is not a valid choice for a metric to be maximised. Please choose one of the following: " +
                        str(metric_to_be_maximised_list))
        logger.error(error_msg)
        settings_valid = False

    # if at least one parameter is wrong the execution stops with an value error
    if not settings_valid:
        raise ValueError()

    # if all settings are valid the program get executed
    logger.info("The chosen settings are valid.")
    return


def import_raw_sensor_data(filedir, filename, separator, header, parse_dates=None, dtype=None, sensor_labels=None):
    """Imports sensor data from csv file.

    :param filedir: The folder the file lies in.
    :param filename: Name of the File.
    :param separator: The character the which saparates each column in a csv file.
    :param header: Indicator at which line the data starts (length of the header)
    :param logging_level: level of logging
    :param parse_dates: Collection of the columns that should get parsed as a date.
    :param dtype: A mapping of data types to the columns in the file.
    :return: the sensor data in a pandas data frame
    """

    # read in the file containing sensor data
    raw_sensor_data = utils.read_csv_file(filedir=filedir, filename=filename, separator=separator, header=header,
                                          parse_dates=parse_dates, dtype=dtype)

    # column to value type assignment with the following pattern: "'<column name in file>': '<internal column type>'"
    column_assignment_sensor_data = {settings.file_column_name_sensor_label: 'SensorLabel',
                                     settings.file_column_name_timestamp: 'DateTime',
                                     settings.file_column_name_status: 'Active'}
    # assign columns to column types
    raw_sensor_data = raw_sensor_data.rename(columns=column_assignment_sensor_data)

    # mapping: sensor label to internal sensor id
    sensor_labels_to_id_mapping = pd.DataFrame({
        'SensorID': range(0, len(sensor_labels)),
        'SensorLabel': sensor_labels
    })
    # add integer sensor id to data
    raw_sensor_data = raw_sensor_data.join(sensor_labels_to_id_mapping.set_index('SensorLabel'),
                                                         on='SensorLabel', how='left')

    # categorize into sensor types by prefix
    raw_sensor_data.loc[raw_sensor_data['SensorLabel'].str.startswith(settings.sensor_type_prefixes['motion']), 'SensorType'] = 'motion'
    raw_sensor_data.loc[raw_sensor_data['SensorLabel'].str.startswith(settings.sensor_type_prefixes['door']), 'SensorType'] = 'door'
    raw_sensor_data.loc[raw_sensor_data['SensorLabel'].str.startswith(settings.sensor_type_prefixes['temperature']), 'SensorType'] = 'temperature'

    # drop all lines without motion sensor (identify by sensor ID-prefix)
    # ToDo: in future versions allow for more sensor types if implemented
    #index_names = raw_sensor_data[raw_sensor_data['SensorLabel'].str.equals('motion')].index
    raw_sensor_data_motion = raw_sensor_data.loc[raw_sensor_data['SensorType'] == 'motion']
    raw_sensor_data_motion.reset_index(inplace=True, drop=True)

    return raw_sensor_data_motion


def param_combination_already_executed(path_data_sources, current_params, dir_export_files, step):
    """
    Checks if the current parameter combination was already executed in previous iterations or other runs. Therefore the
    method checks if the directory at which the export files are saved is already created.

    :param current_params: parameter combination of the current iteration
    :param path_data_sources: directory of data sources
    :param dir_export_files: directory at which the export files are saved
    :param step: the program step in which the parameter combination is executed
    :return: if there are already files and the directory in which the export files are saved
    """
    same_params_executed = False

    # checks if path already exists (creating whole folder path and replacing placeholders)
    dir_same_param = path_data_sources + dir_export_files.format(**current_params)

    # if the directory does not exist the method returns None
    if os.path.exists(dir_same_param):
        same_params_executed = True
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(settings.logging_level)
        logger.info("The current %s parameter combination was already executed in previous iterations or other runs.",
                    step)

    return same_params_executed, dir_same_param


def create_param_opt_space():
    """
    Creates the search space for parameter optimization.
    :return: The space which defines the bounds of the parameters in parameter optimization
    """
    # creates a selection of thresholds out of min, max and threshold step length
    # distance_threshold_list = convert_search_space_into_selection(
    #     search_space_min=settings.distance_threshold_min,
    #     search_space_max=settings.distance_threshold_max,
    #     step_length=settings.distance_threshold_step_length)

    # creates a selection of possible activations per trace out of min, max and step length
    number_of_activations_per_trace_list = convert_search_space_into_selection(
        search_space_min=settings.number_of_activations_per_trace_min,
        search_space_max=settings.number_of_activations_per_trace_max,
        step_length=settings.number_of_activations_per_trace_step_length)

    # creates a selection of trace duration limits out of min, max and step length
    trace_duration_list = convert_search_space_into_selection(
        search_space_min=settings.trace_duration_min,
        search_space_max=settings.trace_duration_max,
        step_length=settings.trace_duration_step_length)

    # creates the search space for "hyperopt" parameter search
    space = {
        'zero_distance_value': hp.randint('zero_distance_value', settings.zero_distance_value_min,
                                          settings.zero_distance_value_max + 1),
        'vectorization_type': hp.choice('vectorization_type', settings.vectorization_type_list),
        'clustering_method': hp.choice('clustering_method', settings.clustering_method_list),
        'trace_partition_method': hp.choice('trace_partition_method', settings.trace_partition_method),
        'number_of_activations_per_trace': hp.choice('number_of_activations_per_trace',
                                                     number_of_activations_per_trace_list),
        'trace_duration': hp.choice('trace_duration', trace_duration_list),
        'hyp_number_of_day_partitions': hp.choice('hyp_number_of_day_partitions',
                                                  settings.hyp_number_of_day_partitions_list),
        'hyp_week_separator': hp.choice('hyp_week_separator', settings.hyp_week_separator_list),
        'hyp_number_of_clusters': hp.randint('hyp_number_of_clusters', settings.hyp_min_number_clusters,
                                             settings.hyp_max_number_clusters + 1)
    }
    return space


def create_som_param_opt_space():
    """
    Creates the search space for som parameter optimization.
    :return: The space which defines the bounds of the parameters in parameter optimization
    """
    space = {
        'lr': hp.uniform('lr', settings.min_lr, settings.max_lr + 0.00000000001),
        'sigma': hp.uniform('sigma', settings.min_sigma, settings.max_sigma + 0.00000000001)
    }
    return space


def show_function_value_history(function_values, iterations):
    """
    Creates a graph that shows the performance history of the program in multiple iterations.
    :param function_values: list of values which define the success of an iteration (precision or fitness)
    :param iterations: list of iteration numbers beginning at 1
    :return:
    """
    plt.clf()
    plt.xlabel('Iteration')
    plt.ylabel(settings.metric_to_be_maximised)
    plt.title(settings.metric_to_be_maximised + ' Value Progress')
    plt.plot(iterations, function_values, marker='o')
    plt.ylim(bottom=0)
    plt.xticks(iterations)
    plt.savefig(settings.path_data_sources + settings.dir_runtime_files + settings.metric_to_be_maximised +
                '_progress.png', dpi=300)
    #plot.show()
    return


def import_user_config():
    """
    Reads in the configuration defined by the user.
    :return:
    """
    config = configparser.ConfigParser()
    config.read(settings.dir_user_config + settings.filename_user_config)

    settings.file_column_name_sensor_label = config['DEFAULT']['file_column_name_sensor_label']
    settings.file_column_name_timestamp = config['DEFAULT']['file_column_name_timestamp']
    settings.file_column_name_status = config['DEFAULT']['file_column_name_status']
    settings.number_of_runs = config.getint('DEFAULT', 'number_of_runs')
    settings.metric_to_be_maximised = config['DEFAULT']['metric_to_be_maximised']
    settings.max_number_of_raw_input = config.getint('DEFAULT', 'max_number_of_raw_input')
    settings.sensor_type_prefixes['motion'] = config['DEFAULT']['motion_sensor_prefix']
    settings.linkage_method_for_clustering = config['DEFAULT']['linkage_method_for_clustering']
    settings.rel_proportion_dfg_threshold = config.getfloat('DEFAULT', 'rel_proportion_dfg_threshold')
    settings.miner_type = config['DEFAULT']['miner_type']
    settings.execution_type = config['DEFAULT']['execution_type']

    settings.trace_partition_method = json.loads(config['hyperopt_params']['trace_partition_method'])
    settings.number_of_activations_per_trace_min = config.getint('hyperopt_params', 'number_of_activations_per_trace_min')
    settings.number_of_activations_per_trace_max = config.getint('hyperopt_params', 'number_of_activations_per_trace_max')
    settings.number_of_activations_per_trace_step_length = config.getint('hyperopt_params', 'number_of_activations_per_trace_step_length')
    settings.trace_duration_min = config.getint('hyperopt_params', 'trace_duration_min')
    settings.trace_duration_max = config.getint('hyperopt_params', 'trace_duration_max')
    settings.trace_duration_step_length = config.getint('hyperopt_params', 'trace_duration_step_length')
    settings.zero_distance_value_min = config.getint('hyperopt_params', 'zero_distance_value_min')
    settings.zero_distance_value_max = config.getint('hyperopt_params', 'zero_distance_value_max')
    settings.hyp_min_number_clusters = config.getint('hyperopt_params', 'hyp_min_number_clusters')
    settings.hyp_max_number_clusters = config.getint('hyperopt_params', 'hyp_max_number_clusters')
    settings.vectorization_type_list = json.loads(config['hyperopt_params']['vectorization_type_list'])
    settings.hyp_number_of_day_partitions_list = json.loads(config['hyperopt_params']['hyp_number_of_day_partitions_list'])
    settings.hyp_week_separator_list = json.loads(config['hyperopt_params']['hyp_week_separator_list'])
    settings.clustering_method_list = json.loads(config['hyperopt_params']['clustering_method_list'])

    settings.fixed_params['zero_distance_value'] = config.getint('fixed_params', 'zero_distance_value')
    settings.fixed_params['traces_time_out_threshold'] = config.getint('fixed_params', 'traces_time_out_threshold')
    settings.fixed_params['trace_length_limit'] = config.getint('fixed_params', 'trace_length_limit')
    settings.fixed_params['custom_distance_number_of_clusters'] = config.getint('fixed_params', 'custom_distance_number_of_clusters')
    settings.fixed_params['distance_threshold'] = config.getfloat('fixed_params', 'distance_threshold')
    settings.fixed_params['max_errors_per_day'] = config.getint('fixed_params', 'max_errors_per_day')
    settings.fixed_params['vectorization_type'] = config['fixed_params']['vectorization_type']
    settings.fixed_params['event_case_correlation_method'] = config['fixed_params']['event_case_correlation_method']
    settings.fixed_params['clustering_method'] = config['fixed_params']['clustering_method']
    settings.fixed_params['hyp_number_of_day_partitions'] = config.getint('fixed_params', 'hyp_number_of_day_partitions')
    settings.fixed_params['hyp_week_separator'] = config['fixed_params']['hyp_week_separator']
    return
