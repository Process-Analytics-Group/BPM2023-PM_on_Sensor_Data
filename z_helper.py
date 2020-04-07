import pandas as pd
import numpy as np
import os
import sys
import inspect
import timeit
from pathlib import Path

import logging

# import settings file
import z_setting_parameters as settings


def create_parameters_log_file(dir_runtime_files,
                               grid_search_parameters):
    # ToDo: add description for each variable
    log_file_content = \
        [['exogenous_path_data_sources', settings.path_data_sources, 'Path of folder with source files'],
         ['exogenous_filename_room_separation', settings.filename_room_separation],
         ['exogenous_filename_adjacency_matrix', settings.filename_adjacency_matrix],
         ['exogenous_filename_LogFile', settings.filename_parameters_file],
         ['exogenous_filename_adjacency_plot', settings.filename_adjacency_plot],
         ['exogenous_dir_name_sensor_data', settings.dir_name_sensor_data],
         ['exogenous_data_types', settings.data_types, 'choose between: quantity, time, quantity_time'],
         ['exogenous_csv_delimiter', settings.csv_delimiter],
         ['exogenous_prefix_motion_sensor_id', settings.prefix_motion_sensor_id],
         ['exogenous_max_number_of_people_in_house', settings.max_number_of_people_in_house],
         ['filename_log_file', settings.filename_log_file + '.log', 'Filename of logfile.'],
         ['exogenous_logging_level', settings.logging_level, 'Lvl of logging for log file and console. (10=Debugging)'],
         ['exogenous_zero_distance_value', grid_search_parameters['zero_distance_value']],
         ['exogenous_distance_threshold', grid_search_parameters['distance_threshold']],
         ['exogenous_traces_time_out_threshold', grid_search_parameters['traces_time_out_threshold']],
         ['exogenous_max_trace_length', grid_search_parameters['max_trace_length']]]

    get_trace = getattr(sys, 'gettrace', None)
    if get_trace() is None:
        log_file_content.append(['debug-mode', 'Deactivated', 'Debug Activated/Deactivated'])
    elif get_trace():
        log_file_content.append(['debug-mode', 'Activated', 'Debug Activated/Deactivated'])

    pandas_log_file_content = pd.DataFrame.from_records(log_file_content, columns=['Variable', 'Value', 'Description'])
    # sort data frame
    pandas_log_file_content = pandas_log_file_content.sort_values(by='Variable')
    target_folder = settings.path_data_sources + dir_runtime_files

    pandas_log_file_content.to_csv(path_or_buf=target_folder + '/' + settings.filename_parameters_file,
                                   index=False,
                                   sep=';')


def append_to_log_file(new_entry_to_log_variable,
                       new_entry_to_log_value,
                       filename_parameters_file,
                       dir_runtime_files,
                       new_entry_to_log_description=None):

    target_file = settings.path_data_sources + dir_runtime_files + '/' + filename_parameters_file
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
                                             list_of_properties):
    # start timer
    t0_runtime = timeit.default_timer()
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])

    folder_name = dir_runtime_files.split('/')[-1]

    benchmark_file_path = Path(path_data_sources + dir_runtime_files.split('/')[0] + '/benchmark.csv')
    pass
    if benchmark_file_path.is_file():
        logger.debug("Reading preexisting benchmark file.")

        pd_benchmark_old = pd.read_csv(benchmark_file_path,
                                       sep=';',
                                       header=0,
                                       index_col=0)
        pd_benchmark_add = pd.DataFrame(list_of_properties, index=[folder_name])
        pd_benchmark_new = pd.concat([pd_benchmark_old, pd_benchmark_add], sort=False)

        pd_benchmark_new.to_csv(benchmark_file_path, sep=';')

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


def read_csv_files():
    """

    :return: Pandas data frame containing the sorted merged sensor information of all files in given folder
    """

    # start timer
    t0_read_csv_files = timeit.default_timer()

    logger = logging.getLogger(inspect.stack()[0][3])

    # # try to find if sensor raw data has been created already.
    # # If yes, use this instead of starting over again
    # raw_data_file = Path(settings.path_data_sources + 'sensor_raw.csv')
    # if raw_data_file.is_file():
    #     logging.info("Reading csv Files from preexisting file '../%s", settings.dir_runtime_files + '/sensor_raw.csv')
    #
    #     # read previously created csv file and import it a
    #     all_data = pd.read_csv(settings.path_data_sources + 'sensor_raw.csv',
    #                            sep=';',
    #                            header=0,
    #                            parse_dates=['DateTime'],
    #                            dtype={'Active': np.int8})
    #     # calculate how many data points there are
    #     number_of_data_points = all_data.shape[0]
    #     # stop timer
    #     t1_read_csv_files = timeit.default_timer()
    #     # calculate runtime
    #     runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)
    #
    #     logging.info("Extracted %s data points from csv-File on disc in %s seconds",
    #                  number_of_data_points, runtime_read_csv_files)
    #     return all_data
    logging.info("Reading csv Files from %s", settings.path_data_sources)

    data = []
    df = pd.DataFrame
    file_count = 0
    for filename in os.listdir(settings.path_data_sources + settings.dir_name_sensor_data + '/'):
        # only look through csv files
        if filename.endswith(".csv"):
            # count number of files
            file_count += 1
            # path of file that should be read
            file_path = settings.path_data_sources + '/' + settings.dir_name_sensor_data + '/' + filename
            # read csv as pandas data-frame
            df = pd.read_csv(filepath_or_buffer=file_path,
                             sep=settings.csv_delimiter,
                             names=['Date', 'Time', 'SensorID', 'Active'],
                             error_bad_lines=False,
                             skip_blank_lines=True,
                             na_filter=False)
        data.append(df)

    # copy all data-frames from data-list into one big pandas data frame
    all_data = pd.concat(data, axis=0, join='outer', ignore_index=False,
                         keys=None, levels=None, names=None, verify_integrity=False,
                         copy=True)

    # only take Motion and Door Sensor Data
    all_data = all_data[all_data['SensorID'].str.contains('M|D')]
    all_data['Active'] = all_data['Active'].replace('ON', 1)
    all_data['Active'] = all_data['Active'].replace('OFF', 0)
    all_data['Active'] = all_data['Active'].replace('OPEN', 1)
    all_data['Active'] = all_data['Active'].replace('CLOSE', 0)

    # concatenate Date and time column
    all_data['DateTime'] = pd.to_datetime(all_data['Date'].map(str) + ' ' + all_data['Time'])
    # delete Time and Date column
    all_data.drop(['Date', 'Time'], axis=1)

    # rearrange columns
    all_data = all_data.reindex(columns=['DateTime', 'SensorID', 'Active'])

    # Convert String to DateTime format
    #all_data['DateTime'] = pd.to_datetime(all_data['DateTime'])

    # Sort rows by DateTime
    all_data.sort_values(by='DateTime', inplace=True)

    # reset index so it matches up after the sorting
    all_data = all_data.reset_index(drop=True)

    # calculate how many data points there are
    number_of_data_points = all_data.shape[0]

    all_data.to_csv(settings.path_data_sources + 'sensor_raw.csv',
                    sep=';',
                    index=None)

    # stop timer
    t1_read_csv_files = timeit.default_timer()
    # calculate runtime
    runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

    logger.info("Extracted %s data points from %s csv-Files in %s seconds",
                number_of_data_points, file_count, runtime_read_csv_files)

    return all_data
