import pandas as pd
import numpy as np
import sys
import inspect
import timeit
from pathlib import Path

import logging

# import settings file
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
         ['exogenous_vectorization_type_list', settings.vectorization_type_list, 'range for vectorization type (parameter optimization)'],
         ['exogenous_prefix_motion_sensor_id', settings.prefix_motion_sensor_id,
          'A word, letter, or number placed before motion sensor number.'],
         ['exogenous_max_number_of_people_in_house', settings.max_number_of_people_in_house,
          'Maximum number of persons which were in the house while the recording of sensor data.'],
         ['exogenous_filename_log_export', settings.filename_log_export, 'Filename of log export file.'],
         ['exogenous_dir_petri_net_files', settings.dir_petri_net_files,
          'Directory in which petri net export files are saved.'],
         ['exogenous_filename_petri_net', settings.filename_petri_net, 'Filename of petri net pnml file.'],
         ['exogenous_dir_dfg_files', settings.dir_dfg_files, 'Directory in which dfg images are saved.'],
         ['exogenous_filename_dfg_cluster', settings.filename_dfg_cluster, 'Filename of dfg cluster image files.'],
         ['exogenous_filename_dfg', settings.filename_dfg, 'Filename of dfg image file.'],
         ['exogenous_rel_proportion_dfg_threshold', settings.rel_proportion_dfg_threshold,
          'Threshold for number of sensor activations at which a sensor is shown in dfg (relative to max occurrences of a sensor).'],
         ['filename_log_file', settings.filename_log_file + '.log', 'Filename of logfile.'],
         ['exogenous_logging_level', settings.logging_level, 'Lvl of logging for log file and console. (10=Debugging)'],
         ['exogenous_zero_distance_value', params['zero_distance_value'],
          'Number representing zero distance to other sensors. (used in creation of distance_matrix_real_world matrix)'],
         ['exogenous_distance_threshold', params['distance_threshold'],
          'Threshold when sensors are considered too far away.'],
         ['exogenous_traces_time_out_threshold', params['traces_time_out_threshold'],
          'The time in seconds in which a sensor activation is assigned to a existing trace.'],
         ['exogenous_trace_length_limit', params['trace_length_limit'],
          'Maximum length of traces. (in case length mode is used to separate raw-traces)']]

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
                       filename_parameters_file,
                       dir_runtime_files,
                       new_entry_to_log_description=None):
    target_file = settings.path_data_sources + dir_runtime_files + filename_parameters_file
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

        logger.info("Saving entry to benchmark file took %s seconds.",
                    runtime)


def create_distance_threshold_list(distance_threshold_min,
                                   distance_threshold_max,
                                   distance_threshold_step_length):
    # creates a list of possible thresholds
    curr_distance_threshold = distance_threshold_min
    distance_threshold_list = []

    while curr_distance_threshold < distance_threshold_max:
        distance_threshold_list.append(curr_distance_threshold)
        curr_distance_threshold += distance_threshold_step_length

    distance_threshold_list.append(distance_threshold_max)

    return distance_threshold_list


def check_settings(zero_distance_value_min, zero_distance_value_max, distance_threshold_min, distance_threshold_max,
                   traces_time_out_threshold_min, traces_time_out_threshold_max, trace_length_limit_min,
                   trace_length_limit_max, k_means_number_of_clusters_min, k_means_number_of_clusters_max,
                   event_case_correlation_method, event_case_correlation_method_list, miner_type, miner_type_list,
                   logging_level):
    # checks settings for correctness (if they are invalid the execution get stopped)
    settings_valid = True

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)

    # checking different settings
    if zero_distance_value_min > zero_distance_value_max:
        logger.error("'zero_distance_value_min' has to be <= 'zero_distance_value_max'")
        settings_valid = False

    if distance_threshold_min > distance_threshold_max:
        logger.error("'distance_threshold_min' has to be <= 'distance_threshold_max'")
        settings_valid = False

    if traces_time_out_threshold_min > traces_time_out_threshold_max:
        logger.error("'traces_time_out_threshold_min' has to be <= 'traces_time_out_threshold_max'")
        settings_valid = False

    if trace_length_limit_min > trace_length_limit_max:
        logger.error("'trace_length_limit_min' has to be <= 'trace_length_limit_max'")
        settings_valid = False

    if k_means_number_of_clusters_min > k_means_number_of_clusters_max:
        logger.error("'k_means_number_of_clusters_min' has to be <= 'k_means_number_of_clusters_max'")
        settings_valid = False

    if event_case_correlation_method not in event_case_correlation_method_list:
        error_msg = str("'" + event_case_correlation_method +
                        "' is not a valid event case correlation method choice. Please choose one of the following: " +
                        str(event_case_correlation_method_list))
        logger.error(error_msg)
        settings_valid = False

    if miner_type not in miner_type_list:
        error_msg = str("'" + miner_type +
                        "' is not a valid choice for a miner. Please choose one of the following: " +
                        str(miner_type_list))
        logger.error(error_msg)
        settings_valid = False

    # if at least one parameter is wrong the execution stops with an value error
    if not settings_valid:
        raise ValueError()

    # if all settings are valid the program get executed
    logger.info("The chosen settings are valid")
    return
