# Personentrennung und Tracelängenbestimmung
import pandas as pd
import numpy as np
import os
import logging
import timeit
from pathlib import Path
import math
import helper
import inspect

def create_trace_from_file(data_sources_path,
                           dict_distance_adjacency_sensor,
                           dir_name_sensor_data,
                           dir_runtime_files,
                           filename_parameters_file,
                           data_types,
                           max_trace_length,
                           max_number_of_raw_input,
                           csv_delimiter=';',
                           prefix_motion_sensor_id='M',
                           distance_threshold=1.5,
                           max_number_of_people_in_house=2,
                           traces_time_out_threshold=300):
    # find all files and consolidate all data in one array
    raw_data = read_csv_files(data_sources_path=data_sources_path,
                              csv_delimiter=csv_delimiter,
                              dir_name_sensor_data=dir_name_sensor_data,
                              filename_parameters_file=filename_parameters_file,
                              dir_runtime_files=dir_runtime_files,
                              max_number_of_raw_input=max_number_of_raw_input)

    # convert_raw_data_to_traces_fast(data_sources_path=data_sources_path,
    #                                 dir_runtime_files=dir_runtime_files,
    #                                 raw_data=raw_data,
    #                                 filename_parameters_file=filename_parameters_file,
    #                                 dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
    #                                 prefix_motion_sensor_id=prefix_motion_sensor_id,
    #                                 distance_threshold=distance_threshold,
    #                                 max_number_of_people_in_house=max_number_of_people_in_house,
    #                                 traces_time_out_threshold=traces_time_out_threshold)

    traces_raw_pd, all_traces_short = \
        convert_raw_data_to_traces(data_sources_path=data_sources_path,
                                   dir_runtime_files=dir_runtime_files,
                                   raw_data=raw_data,
                                   filename_parameters_file=filename_parameters_file,
                                   dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                   prefix_motion_sensor_id=prefix_motion_sensor_id,
                                   distance_threshold=distance_threshold,
                                   max_number_of_people_in_house=max_number_of_people_in_house,
                                   traces_time_out_threshold=traces_time_out_threshold)

    traces_shortened, output_case_traces_cluster, list_of_final_vectors_activations \
        = divide_raw_traces(traces_raw_pd=traces_raw_pd,
                            data_sources_path=data_sources_path,
                            dir_runtime_files=dir_runtime_files,
                            filename_parameters_file=filename_parameters_file,
                            max_trace_length=max_trace_length,
                            data_types=data_types)

    # pairwise_dissimilarity_matrix = calculate_pairwise_dissimilarity(
    #     list_of_final_vectors_activations=list_of_final_vectors_activations,
    #     distance_matrix=dict_distance_adjacency_sensor['distance_matrix'],
    #     data_sources_path=data_sources_path,
    #     dir_runtime_files=dir_runtime_files,
    #     filename_parameters_file=filename_parameters_file,
    #     max_trace_length=max_trace_length)

    return traces_shortened, output_case_traces_cluster, list_of_final_vectors_activations


def calculate_pairwise_dissimilarity(list_of_final_vectors_activations,
                                     distance_matrix,
                                     data_sources_path,
                                     dir_runtime_files,
                                     filename_parameters_file,
                                     max_trace_length
                                     ):
    # start timer
    t0_read_csv_files = timeit.default_timer()
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.debug("Calculating dissimilarity_matrix...")

    i_iterator = 0

    # cut off longer traces or fill shorter traces
    # if one trace shorter, fill with same values as longer, so 0 will be added
    # or maybe fill with 0
    filled_trace_vector = pd.DataFrame(list_of_final_vectors_activations, dtype=int).fillna(0).values
    # convert float to integer
    filled_trace_vector = filled_trace_vector.astype(int)

    # all combinations of the vector with itself
    unsplitted_combination = np.hstack((np.tile(filled_trace_vector, (len(filled_trace_vector), 1)),
                                        np.repeat(filled_trace_vector, len(filled_trace_vector), 0)))

    splitted = np.split(unsplitted_combination, 2, axis=1)

    sum_distance = (distance_matrix[splitted[1], splitted[0]]).sum(1)
    dissimilarity_matrix = np.reshape(sum_distance, (len(list_of_final_vectors_activations),
                                                     len(list_of_final_vectors_activations)))

    # noinspection PyTypeChecker
    np.savetxt(data_sources_path + '/' + dir_runtime_files + '/' + 'pairwise_dissimilarity_matrix.csv',
               X=dissimilarity_matrix,
               fmt='%i',
               delimiter=";")
    # stop timer
    t1_read_csv_files = timeit.default_timer()
    # calculate runtime
    runtime_dissimilarity_matrix = np.round(t1_read_csv_files - t0_read_csv_files, 1)

    helper.append_to_log_file(
        new_entry_to_log_variable='runtime_' + inspect.stack()[0][3],
        new_entry_to_log_value=runtime_dissimilarity_matrix,
        path_data_sources=data_sources_path,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Seconds it took to calculate dissimilarity matrix.')

    logger.info("Calculating the dissimilarity matrix took %s seconds.", runtime_dissimilarity_matrix)
    return dissimilarity_matrix


def divide_raw_traces(traces_raw_pd,
                      data_sources_path,
                      dir_runtime_files,
                      filename_parameters_file,
                      max_trace_length,
                      data_types):
    # start timer
    t0_read_csv_files = timeit.default_timer()
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    # ToDo ordentlicher machen
    list_of_final_vectors = []
    list_of_final_vectors_quantity = []
    list_of_activations = []
    list_of_final_vectors_activations = []
    list_header = [list(range(52))]
    list_header[0].insert(0, 'Case')
    length_count = 1
    final_vector_time = np.zeros(53)
    final_vector_quantity = np.zeros(53, dtype=int)
    case = 1
    unique_identifier = 1
    case_id_for_short_traces = []

    for data_row in traces_raw_pd.itertuples():

        if case != data_row.Case or length_count > max_trace_length:
            final_vector_time[0] = unique_identifier
            final_vector_quantity[0] = unique_identifier
            list_of_final_vectors.append(final_vector_time)
            list_of_final_vectors_quantity.append(final_vector_quantity)
            list_of_final_vectors_activations.append(np.array(list_of_activations))
            unique_identifier += 1
            length_count = 1
            final_vector_time = np.zeros(53)
            final_vector_quantity = np.zeros(53, dtype=int)
            list_of_activations = []
        case_id_for_short_traces.append(unique_identifier)
        case = data_row.Case
        # add +1 to vector that counts quantity
        list_of_activations.append(data_row.Sensor_Added)
        try:
            final_vector_quantity[data_row.Activity[-1] + 1] += 1
        except IndexError:
            # if no sensor activated, add quantity to M0
            final_vector_quantity[1] += 1
        # if no entry, M0 is active
        if len(data_row.Activity) == 0:
            if math.isnan(data_row.Duration):
                final_vector_time[1] += 0
            else:
                final_vector_time[1] += data_row.Duration
        for sensor_id in data_row.Activity:
            final_vector_time[int(sensor_id) + 1] += data_row.Duration
        length_count += 1
    # take last unfinished trace and copy it too
    final_vector_time[0] = unique_identifier
    final_vector_quantity[0] = unique_identifier
    list_of_final_vectors.append(final_vector_time)
    list_of_final_vectors_quantity.append(final_vector_quantity)
    list_of_final_vectors_activations.append(np.array(list_of_activations))

    final_vector_time = pd.DataFrame(list_of_final_vectors, columns=list_header[0])
    final_vector_quantity = pd.DataFrame(list_of_final_vectors_quantity, columns=list_header[0])
    final_vector_time.Case = final_vector_time.Case.astype(int)

    # read traces_raw
    output_case_traces_cluster = pd.read_csv(data_sources_path + dir_runtime_files + '/traces_raw.csv',
                                             sep=';',
                                             header=0)

    output_case_traces_cluster['Person'] = output_case_traces_cluster['Case']
    output_case_traces_cluster['Case'] = case_id_for_short_traces

    if data_types == 'quantity':
        final_vector = final_vector_quantity
    elif data_types == 'time':
        final_vector = final_vector_time
    elif data_types == 'quantity_time':
        # drop caseID of second dataframe
        final_vector_quantity = final_vector_quantity[final_vector_quantity.columns[1:]]
        final_vector = pd.concat([final_vector_time, final_vector_quantity],
                                 axis=1,
                                 ignore_index=True)
    else:
        logger.exception(data_types + ' is not a valid choice.[quantity, time, quantity_time]')

    # write traces to disk
    final_vector.to_csv(data_sources_path + dir_runtime_files + '/traces_basic.csv',
                        sep=';',
                        index=None)

    # stop timer
    t1_read_csv_files = timeit.default_timer()
    # calculate runtime
    runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

    helper.append_to_log_file(
        new_entry_to_log_variable='case_id_for_short_traces',
        new_entry_to_log_value=unique_identifier - 1,
        path_data_sources=data_sources_path,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Number of shorter traces of length ' + str(max_trace_length) + '.')

    helper.append_to_log_file(
        new_entry_to_log_variable='runtime_' + inspect.stack()[0][3],
        new_entry_to_log_value=runtime_read_csv_files,
        path_data_sources=data_sources_path,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Seconds it took to divide long traces into shorter traces.')

    logger.info("Dividing the long traces into %s short traces took %s seconds.",
                unique_identifier - 1, runtime_read_csv_files)
    return final_vector, output_case_traces_cluster, list_of_final_vectors_activations


def convert_raw_data_to_traces_fast(raw_data,
                                    dict_distance_adjacency_sensor,
                                    data_sources_path,
                                    dir_runtime_files,
                                    filename_parameters_file,
                                    prefix_motion_sensor_id='M',
                                    distance_threshold=1.5,
                                    max_number_of_people_in_house=2,
                                    traces_time_out_threshold=300):
    # start timer
    t0_convert_raw2trace = timeit.default_timer()

    # get distance matrix from dictionary
    distance_matrix = dict_distance_adjacency_sensor['distance_matrix']
    np_raw_data = raw_data.values

    trace_open = {}
    unique_new_trace_id = 1

    for data_row in np_raw_data:
        # if not a motion sensor, continue with next row
        # ToDo: Maybe also accept door sensors for main entries
        if data_row[1][0:len(prefix_motion_sensor_id)] != prefix_motion_sensor_id:
            continue

        # extract sensor ID and skip prefix:
        # ToDo: Do this for all elements of that column under the condition of it being an M Sensor
        sensor_id_no_prefix = int(data_row[1][len(prefix_motion_sensor_id):])

        # if sensor is greater than 51, the highest sensor id, skip and report
        # ToDo: Do this not with a fixed value, but with the upper limit being variable (use dim of distance matrix)
        if sensor_id_no_prefix > 51:
            # logging.warning("Found sensor M%s in row %s is not part of the real world layout. It has been skipped.",
            #                sensor_id_no_prefix, data_row)
            continue
        ##
        active_trace_id_list = []
        # close open traces
        # CASE: Time-Out (Trace has been inactive longer than a preset threshold)
        traces_to_be_closed = []
        for trace_id, value in trace_open.items():
            # if no Sensor active and all open traces are empty, skip row
            if not trace_open[trace_id]:
                continue
            activation_time = np.round((data_row.DateTime - trace_open[trace_id][-1][2]) / np.timedelta64(1, 's'), 2)
            # only close if no sensor currently activated
            if activation_time > traces_time_out_threshold and trace_open[trace_id][-1][0] == []:
                traces_to_be_closed.append(trace_id)

        for trace_to_close in traces_to_be_closed:
            # remove one of the timestamps of last entry
            trace_open[trace_to_close][-1][2] = None
            temporary_df = pd.DataFrame(columns=['Activity', 'Sensor_Added', 'Duration',
                                                 'Timestamp', 'LC_Activity', 'LC'])
            temporary_df = temporary_df.from_records(data=trace_open[trace_to_close],
                                                     columns=['Activity', 'Sensor_Added', 'Duration',
                                                              'Timestamp', 'LC_Activity', 'LC'])
            temporary_df['Case'] = trace_to_close
            # get activated-sensors from data frame to a list
            added_sensors_list = temporary_df['Sensor_Added'].values.tolist()
            pd_df_all_traces_short.loc[trace_to_close] = [trace_to_close, added_sensors_list]
            # remove Sensor_Added Column
            # temporary_df.drop('Sensor_Added', axis=1, inplace=True)
            # todo: Move to end and do them all at once to save time
            pd_df_all_traces = pd.concat([pd_df_all_traces, temporary_df], axis=0, join='outer', join_axes=None,
                                         ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False,
                                         copy=True, sort=False)
            # remove extracted list from open traces
            trace_open.pop(trace_to_close, None)

        # check if new sensor is in proximity of previously activated sensors
        # loop through trace_open to check in all open traces
        # dictionary with the average distances from the sensor to the last entry of all active traces
        if trace_open:
            if data_row.Active == 1:
                average_distance_dict = {}
                for trace_id, value in trace_open.items():
                    skip_last_timestamp = 0
                    sum_of_distances = 0
                    # only look into last entry of every dict
                    # loop through all activated sensors in last entry
                    # if last time stamp no sensors have been activated, look into second last time stamp
                    if not trace_open[trace_id][-1][0]:
                        skip_last_timestamp = 1
                    weighted_decay_factor = 1
                    for sensor_element in trace_open[trace_id][-1 - skip_last_timestamp][0]:
                        sum_of_distances += weighted_decay_factor * distance_matrix[sensor_element][sensor_id_no_prefix]
                        weighted_decay_factor += 1
                    # calculate weighted average distance with decay function to active sensors
                    # sensors activated a while ago have less influence than recent sensors
                    average_distance = sum_of_distances / sum(range(weighted_decay_factor))
                    # add average distance to dictionary with average distance as key
                    if average_distance in average_distance_dict:
                        average_distance_dict[average_distance].append(trace_id)
                    else:
                        average_distance_dict[average_distance] = [trace_id]
                # if average distance is over a threshold, it is considered too far away from currently active trace
                # only consider a maximum amount of people in the house at the same time
                # ToDo: find a better solution for the max amount of people
                if min(average_distance_dict) <= distance_threshold or len(trace_open) >= max_number_of_people_in_house:
                    # get trace-ID(s) with the lowest average distance
                    active_trace_id_list = average_distance_dict[min(average_distance_dict)]
                else:
                    trace_open[unique_new_trace_id] = []
                    active_trace_id_list.append(unique_new_trace_id)
                    unique_new_trace_id += 1

            elif data_row.Active == 0:
                # mark all traces that contain deactivated sensor
                for trace_id, value in trace_open.items():
                    if trace_open[trace_id] and sensor_id_no_prefix in trace_open[trace_id][-1][0]:
                        active_trace_id_list.append(trace_id)
        # if traces open is still empty, create new personalised trace
        elif data_row.Active == 1:
            trace_open[unique_new_trace_id] = []
            # identifier which trace to write to
            active_trace_id_list = [unique_new_trace_id]
            unique_new_trace_id += 1
        # if more than one trace is suitable add to all of them
        for active_trace_id in active_trace_id_list:
            # CASE: if active_trace_id is empty AND Sensor is Activated
            if not trace_open[active_trace_id] and data_row.Active == 1:
                trace_open[active_trace_id] = [[[sensor_id_no_prefix],
                                                sensor_id_no_prefix,
                                                data_row.DateTime,
                                                data_row.DateTime,
                                                int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                's']]
            elif not trace_open[active_trace_id] and data_row.Active == 0:
                continue
            # CASE: if active_trace_id exists
            else:

                # calculate activation time:
                activation_time = np.round((data_row.DateTime - trace_open[active_trace_id][-1][2])
                                           / np.timedelta64(1, 's'), 2)
                # CASE: Active
                if data_row.Active == 1:
                    # add sensor to active trace
                    # modify the time of last entry so it matches the duration it was active
                    trace_open[active_trace_id][-1][2] = activation_time

                    # append currently active sensor to sensor list
                    new_sensor_activation_list = trace_open[active_trace_id][-1][0] + [sensor_id_no_prefix]

                    trace_open[active_trace_id].append([new_sensor_activation_list,
                                                        sensor_id_no_prefix,
                                                        data_row.DateTime,
                                                        data_row.DateTime,
                                                        int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                        's'])
                # CASE: Deactivated,
                # delete from active list
                elif data_row.Active == 0:
                    # copy active sensors from last entry
                    new_sensor_activation_list = trace_open[active_trace_id][-1][0][:]
                    # remove now deactivated sensor
                    new_sensor_activation_list.remove(int(data_row.SensorID[len(prefix_motion_sensor_id):]))

                    # modify the time of last entry so it matches the duration it was active
                    trace_open[active_trace_id][-1][2] = activation_time
                    # differentiate for the 'last activated' column
                    if new_sensor_activation_list:
                        trace_open[active_trace_id].append([new_sensor_activation_list,
                                                            None,
                                                            data_row.DateTime,
                                                            data_row.DateTime,
                                                            int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                            'c'])
                    # if no sensor is activated anymore, document that in 'last activated' column
                    else:
                        trace_open[active_trace_id].append([new_sensor_activation_list,
                                                            None,
                                                            data_row.DateTime,
                                                            data_row.DateTime,
                                                            int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                            'c'])

    print()


def convert_raw_data_to_traces(raw_data,
                               dict_distance_adjacency_sensor,
                               data_sources_path,
                               dir_runtime_files,
                               filename_parameters_file,
                               prefix_motion_sensor_id='M',
                               distance_threshold=1.5,
                               max_number_of_people_in_house=2,
                               traces_time_out_threshold=300):
    # start timer
    t0_read_csv_files = timeit.default_timer()
    logger = logging.getLogger(inspect.stack()[0][3])
    # try to find if sensor raw data has been created already.
    # If yes, use this instead of starting over again
    pd_df_all_traces_short_file = Path(data_sources_path + dir_runtime_files + '/traces_raw_short.csv')
    pd_df_all_traces_file = Path(data_sources_path + dir_runtime_files + '/traces_raw.csv')
    if pd_df_all_traces_short_file.is_file() and pd_df_all_traces_file.is_file():
        logging.info("Reading csv Files from preexisting file '../%s", dir_runtime_files + '/traces_raw_short.csv')
        logging.info("Reading csv Files from preexisting file '../%s", dir_runtime_files + '/traces_raw.csv')
        # read previously created csv file and import it a
        pd_df_all_traces = pd.read_csv(pd_df_all_traces_file,
                                       sep=';',
                                       header=0)
        pd_df_all_traces_short = pd.read_csv(pd_df_all_traces_short_file,
                                             sep=';',
                                             header=0)
        # calculate how many data points there are
        number_of_data_points = pd_df_all_traces.shape[0]
        # stop timer
        t1_read_csv_files = timeit.default_timer()
        # calculate runtime
        runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

        logger.info("Extracted %s data points from csv-File on disc in %s seconds",
                    number_of_data_points, runtime_read_csv_files)
        return pd_df_all_traces, pd_df_all_traces_short

    # get distance matrix from dictionary
    distance_matrix = dict_distance_adjacency_sensor['distance_matrix']

    # create empty pandas Data Frame where all traces are saved with their Trace-ID
    pd_df_all_traces = pd.DataFrame(columns=['Case', 'Activity', 'Duration', 'Timestamp'])

    pd_df_all_traces_short = pd.DataFrame(columns=['Case', 'Activity'])

    # currently filled traces (not yet closed)
    trace_open = {}
    unique_new_trace_id = 1
    for data_row in raw_data.itertuples():
        # if not a motion sensor, continue with next row
        if data_row.SensorID[0:len(prefix_motion_sensor_id)] != prefix_motion_sensor_id:
            continue

        # extract sensor ID and skip prefix:
        sensor_id_no_prefix = int(data_row.SensorID[len(prefix_motion_sensor_id):])
        # if sensor is greater than 51, the highest sensor id, skip and report
        if sensor_id_no_prefix > 51:
            # logging.warning("Found sensor M%s in row %s is not part of the real world layout. It has been skipped.",
            #                sensor_id_no_prefix, data_row)
            continue
        active_trace_id_list = []
        # close open traces
        # CASE: Time-Out (Trace has been inactive longer than a preset threshold)
        traces_to_be_closed = []
        for trace_id, value in trace_open.items():
            # if no Sensor active and all open traces are empty, skip row
            if not trace_open[trace_id]:
                continue
            activation_time = np.round((data_row.DateTime - trace_open[trace_id][-1][2]) / np.timedelta64(1, 's'), 2)
            # only close if no sensor currently activated
            if activation_time > traces_time_out_threshold and trace_open[trace_id][-1][0] == []:
                traces_to_be_closed.append(trace_id)

        for trace_to_close in traces_to_be_closed:
            # remove one of the timestamps of last entry
            trace_open[trace_to_close][-1][2] = None
            temporary_df = pd.DataFrame(columns=['Activity', 'Sensor_Added', 'Duration',
                                                 'Timestamp', 'LC_Activity', 'LC'])
            temporary_df = temporary_df.from_records(data=trace_open[trace_to_close],
                                                     columns=['Activity', 'Sensor_Added', 'Duration',
                                                              'Timestamp', 'LC_Activity', 'LC'])
            temporary_df['Case'] = trace_to_close
            # get activated-sensors from data frame to a list
            added_sensors_list = temporary_df['Sensor_Added'].values.tolist()
            pd_df_all_traces_short.loc[trace_to_close] = [trace_to_close, added_sensors_list]
            # remove Sensor_Added Column
            # temporary_df.drop('Sensor_Added', axis=1, inplace=True)
            # todo: Move to end and do them all at once to save time
            pd_df_all_traces = pd.concat([pd_df_all_traces, temporary_df], axis=0, join='outer', join_axes=None,
                                         ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False,
                                         copy=True, sort=False)
            # remove extracted list from open traces
            trace_open.pop(trace_to_close, None)

        # check if new sensor is in proximity of previously activated sensors
        # loop through trace_open to check in all open traces
        # dictionary with the average distances from the sensor to the last entry of all active traces
        if trace_open:
            if data_row.Active == 1:
                average_distance_dict = {}
                for trace_id, value in trace_open.items():
                    skip_last_timestamp = 0
                    sum_of_distances = 0
                    # only look into last entry of every dict
                    # loop through all activated sensors in last entry
                    # if last time stamp no sensors have been activated, look into second last time stamp
                    if not trace_open[trace_id][-1][0]:
                        skip_last_timestamp = 1
                    weighted_decay_factor = 1
                    for sensor_element in trace_open[trace_id][-1 - skip_last_timestamp][0]:
                        sum_of_distances += weighted_decay_factor * distance_matrix[sensor_element][sensor_id_no_prefix]
                        weighted_decay_factor += 1
                    # calculate weighted average distance with decay function to active sensors
                    # sensors activated a while ago have less influence than recent sensors
                    average_distance = sum_of_distances / sum(range(weighted_decay_factor))
                    # add average distance to dictionary with average distance as key
                    if average_distance in average_distance_dict:
                        average_distance_dict[average_distance].append(trace_id)
                    else:
                        average_distance_dict[average_distance] = [trace_id]
                # if average distance is over a threshold, it is considered too far away from currently active trace
                # only consider a maximum amount of people in the house at the same time
                # ToDo: find a better solution for the max amount of people
                if min(average_distance_dict) <= distance_threshold or len(trace_open) >= max_number_of_people_in_house:
                    # get trace-ID(s) with the lowest average distance
                    active_trace_id_list = average_distance_dict[min(average_distance_dict)]
                else:
                    trace_open[unique_new_trace_id] = []
                    active_trace_id_list.append(unique_new_trace_id)
                    unique_new_trace_id += 1

            elif data_row.Active == 0:
                # mark all traces that contain deactivated sensor
                for trace_id, value in trace_open.items():
                    if trace_open[trace_id] and sensor_id_no_prefix in trace_open[trace_id][-1][0]:
                        active_trace_id_list.append(trace_id)
        # if traces open is still empty, create new personalised trace
        elif data_row.Active == 1:
            trace_open[unique_new_trace_id] = []
            # identifier which trace to write to
            active_trace_id_list = [unique_new_trace_id]
            unique_new_trace_id += 1
        # if more than one trace is suitable add to all of them
        for active_trace_id in active_trace_id_list:
            # CASE: if active_trace_id is empty AND Sensor is Activated
            if not trace_open[active_trace_id] and data_row.Active == 1:
                trace_open[active_trace_id] = [[[sensor_id_no_prefix],
                                                sensor_id_no_prefix,
                                                data_row.DateTime,
                                                data_row.DateTime,
                                                int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                's']]
            elif not trace_open[active_trace_id] and data_row.Active == 0:
                continue
            # CASE: if active_trace_id exists
            else:

                # calculate activation time:
                activation_time = np.round((data_row.DateTime - trace_open[active_trace_id][-1][2])
                                           / np.timedelta64(1, 's'), 2)
                # CASE: Active
                if data_row.Active == 1:
                    # add sensor to active trace
                    # modify the time of last entry so it matches the duration it was active
                    trace_open[active_trace_id][-1][2] = activation_time

                    # append currently active sensor to sensor list
                    new_sensor_activation_list = trace_open[active_trace_id][-1][0] + [sensor_id_no_prefix]

                    trace_open[active_trace_id].append([new_sensor_activation_list,
                                                        sensor_id_no_prefix,
                                                        data_row.DateTime,
                                                        data_row.DateTime,
                                                        int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                        's'])
                # CASE: Deactivated,
                # delete from active list
                elif data_row.Active == 0:
                    # copy active sensors from last entry
                    new_sensor_activation_list = trace_open[active_trace_id][-1][0][:]
                    # remove now deactivated sensor
                    new_sensor_activation_list.remove(int(data_row.SensorID[len(prefix_motion_sensor_id):]))

                    # modify the time of last entry so it matches the duration it was active
                    trace_open[active_trace_id][-1][2] = activation_time
                    # differentiate for the 'last activated' column
                    if new_sensor_activation_list:
                        trace_open[active_trace_id].append([new_sensor_activation_list,
                                                            None,
                                                            data_row.DateTime,
                                                            data_row.DateTime,
                                                            int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                            'c'])
                    # if no sensor is activated anymore, document that in 'last activated' column
                    else:
                        trace_open[active_trace_id].append([new_sensor_activation_list,
                                                            None,
                                                            data_row.DateTime,
                                                            data_row.DateTime,
                                                            int(data_row.SensorID[len(prefix_motion_sensor_id):]),
                                                            'c'])

    # rearrange columns
    pd_df_all_traces = pd_df_all_traces.reindex(columns=['Case', 'Activity', 'Sensor_Added', 'Duration',
                                                         'Timestamp', 'LC_Activity', 'LC'])

    # replace 'nan' with 0
    pd_df_all_traces['Sensor_Added'].fillna(0, inplace=True)
    # Convert Float to integer format
    pd_df_all_traces['Sensor_Added'] = pd.to_numeric(pd_df_all_traces['Sensor_Added'], downcast='signed')
    pd_df_all_traces['LC_Activity'] = pd.to_numeric(pd_df_all_traces['LC_Activity'], downcast='signed')
    # sorting
    pd_df_all_traces = pd_df_all_traces.sort_values(by=['Case', 'Timestamp'])
    pd_df_all_traces_short = pd_df_all_traces_short.sort_values(by=['Case'])

    # reset index so it matches up after the sorting
    pd_df_all_traces = pd_df_all_traces.reset_index(drop=True)
    pd_df_all_traces_short = pd_df_all_traces_short.reset_index(drop=True)

    # write traces to disk
    pd_df_all_traces.to_csv(data_sources_path + dir_runtime_files + '/traces_raw.csv',
                            sep=';',
                            index=None)
    pd_df_all_traces_short.to_csv(data_sources_path + dir_runtime_files + '/traces_raw_short.csv',
                                  sep=';',
                                  index=None)
    # stop timer
    t1_read_csv_files = timeit.default_timer()
    # calculate runtime
    runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

    helper.append_to_log_file(new_entry_to_log_variable='runtime_' + inspect.stack()[0][3],
                              new_entry_to_log_value=runtime_read_csv_files,
                              path_data_sources=data_sources_path,
                              dir_runtime_files=dir_runtime_files,
                              filename_parameters_file=filename_parameters_file,
                              new_entry_to_log_description='Seconds it took to extract the traces.')

    helper.append_to_log_file(new_entry_to_log_variable='unique_new_trace_id',
                              new_entry_to_log_value=unique_new_trace_id - 1,
                              path_data_sources=data_sources_path,
                              dir_runtime_files=dir_runtime_files,
                              filename_parameters_file=filename_parameters_file,
                              new_entry_to_log_description='Number of Traces in the long format')

    logging.info("Extracting %s traces took %s seconds.",
                 unique_new_trace_id - 1, runtime_read_csv_files)

    return pd_df_all_traces, pd_df_all_traces_short


def read_csv_files(data_sources_path,
                   dir_name_sensor_data,
                   dir_runtime_files,
                   filename_parameters_file,
                   max_number_of_raw_input,
                   csv_delimiter=';'):
    """
    :param data_sources_path: Path where the source files can be found
    :param dir_name_sensor_data: Name of the folder where the sensor data is located
    :param csv_delimiter: Char that separates columns in csv files
    :return: Pandas data frame containing the sorted merged sensor information of all files in given folder
    """

    # start timer
    t0_read_csv_files = timeit.default_timer()

    logger = logging.getLogger(inspect.stack()[0][3])

    # try to find if sensor raw data has been created already.
    # If yes, use this instead of starting over again
    raw_data_file = Path(data_sources_path + dir_runtime_files + '/sensor_raw.csv')
    if raw_data_file.is_file():
        logging.info("Reading csv Files from preexisting file '../%s", dir_runtime_files + '/sensor_raw.csv')

        # read previously created csv file and import it a
        all_data = pd.read_csv(data_sources_path + dir_runtime_files + '/sensor_raw.csv',
                               sep=';',
                               header=0,
                               parse_dates=['DateTime'],
                               dtype={'Active': np.int8})
        # calculate how many data points there are
        number_of_data_points = all_data.shape[0]
        # stop timer
        t1_read_csv_files = timeit.default_timer()
        # calculate runtime
        runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

        logging.info("Extracted %s data points from csv-File on disc in %s seconds",
                     number_of_data_points, runtime_read_csv_files)
        return all_data
    logging.info("Reading csv Files from %s", data_sources_path)

    data = []
    df = pd.DataFrame
    file_count = 0
    for filename in os.listdir(data_sources_path + dir_name_sensor_data + '/'):
        # only look through csv files
        if filename.endswith(".csv"):
            # count number of files
            file_count += 1
            # path of file that should be read
            file_path = data_sources_path + '/' + dir_name_sensor_data + '/' + filename
            # read csv as pandas data-frame
            df = pd.read_csv(filepath_or_buffer=file_path,
                             sep=csv_delimiter,
                             names=['Date', 'Time', 'SensorID', 'Active'],
                             error_bad_lines=False,
                             skip_blank_lines=True,
                             na_filter=False)
        data.append(df)

    # copy all data-frames from data-list into one big pandas data frame
    all_data = pd.concat(data, axis=0, join='outer', join_axes=None, ignore_index=False,
                         keys=None, levels=None, names=None, verify_integrity=False,
                         copy=True)

    # only take Motion and Door Sensor Data
    all_data = all_data[all_data['SensorID'].str.contains('M|D')]
    all_data['Active'] = all_data['Active'].replace('ON', 1)
    all_data['Active'] = all_data['Active'].replace('OFF', 0)
    all_data['Active'] = all_data['Active'].replace('OPEN', 1)
    all_data['Active'] = all_data['Active'].replace('CLOSE', 0)

    # concatenate Date and time column
    all_data['DateTime'] = all_data['Date'].map(str) + ' ' + all_data['Time']

    # delete Time and Date column
    all_data.drop(['Date', 'Time'], axis=1)

    # rearrange columns
    all_data = all_data.reindex(columns=['DateTime', 'SensorID', 'Active'])

    # Convert String to DateTime format
    all_data['DateTime'] = pd.to_datetime(all_data['DateTime'])

    # Sort rows by DateTime
    all_data.sort_values(by='DateTime', inplace=True)

    # reset index so it matches up after the sorting
    all_data = all_data.reset_index(drop=True)

    number_of_data_points = all_data.shape[0]
    # if number of data points is lower than max number of raw input
    if number_of_data_points < max_number_of_raw_input:
        max_number_of_raw_input = number_of_data_points
    all_data = all_data.head(max_number_of_raw_input)

    number_of_data_points = all_data.shape[0]
    # calculate how many data points there are
    number_of_data_points = all_data.shape[0]

    all_data.to_csv(data_sources_path + dir_runtime_files + '/sensor_raw.csv',
                    sep=';',
                    index=None)

    helper.append_to_log_file(new_entry_to_log_variable='number_of_data_points',
                              new_entry_to_log_value=number_of_data_points,
                              path_data_sources=data_sources_path,
                              dir_runtime_files=dir_runtime_files,
                              filename_parameters_file=filename_parameters_file,
                              new_entry_to_log_description='Number of extracted data points from input file.')

    # stop timer
    t1_read_csv_files = timeit.default_timer()
    # calculate runtime
    runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

    logger.info("Extracted %s data points from %s csv-Files in %s seconds",
                number_of_data_points, file_count, runtime_read_csv_files)

    return all_data