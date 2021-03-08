# Personentrennung und Tracelängenbestimmung
import pandas as pd
import numpy as np
import logging
import timeit
import math
import z_helper
import z_utils as utils
import inspect


def create_trace_from_file(data_sources_path,
                           dict_distance_adjacency_sensor,
                           filename_sensor_data,
                           rel_dir_name_sensor_data,
                           csv_delimiter_sensor_data,
                           csv_header_sensor_data,
                           csv_parse_dates_sensor_data,
                           csv_dtype_sensor_data,
                           filename_traces_raw,
                           csv_delimiter_traces,
                           filename_traces_basic,
                           csv_delimiter_traces_basic,
                           dir_runtime_files,
                           filename_parameters_file,
                           data_types,
                           data_types_list,
                           trace_length_limit,
                           max_number_of_raw_input,
                           logging_level,
                           number_of_motion_sensors,
                           prefix_motion_sensor_id='M',
                           distance_threshold=1.5,
                           max_number_of_people_in_house=2,
                           traces_time_out_threshold=300):
    """
    Creates traces out of a file with sensor activations.

    :param data_sources_path: path of sources
    :param dict_distance_adjacency_sensor: dictionary which contains a matrix which determine which sensors are near to each other
    :param filename_sensor_data: filename of the file that contains the sensor data
    :param rel_dir_name_sensor_data: folder name containing sensor data (relative from directory of sources)
    :param csv_delimiter_sensor_data: delimiter of the columns in csv file of sensor data (input)
    :param csv_header_sensor_data: indicator at which line the data starts
    :param csv_parse_dates_sensor_data: columns that should get parsed as a date
    :param csv_dtype_sensor_data: assignment of data types of columns in the file
    :param filename_traces_raw: filename of trace file
    :param csv_delimiter_traces: csv delimiter of trace files
    :param filename_traces_basic: filename of divided trace file
    :param csv_delimiter_traces_basic: csv delimiter of divided trace file
    :param dir_runtime_files: the folder of the current run
    :param filename_parameters_file: filename of parameters file
    :param data_types: column/s which is/are applied to divided traces
    :param data_types_list: collection of all available data types
    :param trace_length_limit: maximum length of traces (in case length mode is used to separate raw-traces)
    :param max_number_of_raw_input: limitation of the number of processed activations
    :param logging_level: level of logging
    :param number_of_motion_sensors: number of motion sensors
    :param prefix_motion_sensor_id: prefix of motion sensor IDs
    :param distance_threshold: threshold when sensors are considered too far away
    :param max_number_of_people_in_house: maximum number of persons which were in the house while the recording of sensor data
    :param traces_time_out_threshold: the time in seconds in which a sensor activation is assigned to a existing trace
    :return: The divided traces (traces_shortened), raw traces clustered into cases (output_case_traces_cluster),
    activated sensors grouped by divided traces (list_of_final_vectors_activations)
    """

    # read in the sensor data as a pandas data frame
    raw_sensor_data = read_in_sensor_data(data_sources_path=data_sources_path,
                                          filename_sensor_data=filename_sensor_data,
                                          rel_dir_name_sensor_data=rel_dir_name_sensor_data,
                                          csv_delimiter_sensor_data=csv_delimiter_sensor_data,
                                          csv_header_sensor_data=csv_header_sensor_data,
                                          csv_parse_dates_sensor_data=csv_parse_dates_sensor_data,
                                          csv_dtype_sensor_data=csv_dtype_sensor_data,
                                          logging_level=logging_level)

    # limits the number of data points by max_number_of_raw_input
    raw_sensor_data = limit_raw_sensor_data_points(raw_sensor_data=raw_sensor_data,
                                                   max_number_of_raw_input=max_number_of_raw_input,
                                                   logging_level=logging_level)

    # creates traces out of the raw data
    traces_raw_pd = convert_raw_data_to_traces(data_sources_path=data_sources_path,
                                               dir_runtime_files=dir_runtime_files,
                                               raw_sensor_data=raw_sensor_data,
                                               filename_parameters_file=filename_parameters_file,
                                               filename_traces_raw=filename_traces_raw,
                                               csv_delimiter_traces=csv_delimiter_traces,
                                               dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                               logging_level=logging_level,
                                               number_of_motion_sensors=number_of_motion_sensors,
                                               prefix_motion_sensor_id=prefix_motion_sensor_id,
                                               distance_threshold=distance_threshold,
                                               max_number_of_people_in_house=max_number_of_people_in_house,
                                               traces_time_out_threshold=traces_time_out_threshold)

    # split up the raw traces to shorter traces
    traces_shortened, output_case_traces_cluster, list_of_final_vectors_activations \
        = divide_raw_traces(traces_raw_pd=traces_raw_pd,
                            data_sources_path=data_sources_path,
                            filename_traces_basic=filename_traces_basic,
                            csv_delimiter_traces_basic=csv_delimiter_traces_basic,
                            dir_runtime_files=dir_runtime_files,
                            filename_parameters_file=filename_parameters_file,
                            max_trace_length=trace_length_limit,
                            data_types=data_types,
                            data_types_list=data_types_list,
                            number_of_motion_sensors=number_of_motion_sensors,
                            logging_level=logging_level)

    # pairwise_dissimilarity_matrix = calculate_pairwise_dissimilarity(
    #     list_of_final_vectors_activations=list_of_final_vectors_activations,
    #     distance_matrix=dict_distance_adjacency_sensor['distance_matrix'],
    #     data_sources_path=data_sources_path,
    #     dir_runtime_files=dir_runtime_files,
    #     filename_parameters_file=filename_parameters_file,
    #     max_trace_length=max_trace_length)
    output_case_traces_cluster['Timestamp'] = pd.to_datetime(output_case_traces_cluster['Timestamp'])
    return traces_shortened, output_case_traces_cluster, list_of_final_vectors_activations


def limit_raw_sensor_data_points(raw_sensor_data, max_number_of_raw_input, logging_level):
    """
    Limits the number of sensor points to be processed.

    :param raw_sensor_data: the sensor data points
    :param max_number_of_raw_input: limit for number of sensor activations
    :param logging_level: level of logging
    :return: the sensor data points limited by max_number_of_raw_input
    """
    number_of_data_points = raw_sensor_data.shape[0]
    # number sensor data points is only limited if a limit is set and the limit is lower than the actual number of
    # sensor data points
    if max_number_of_raw_input is not None and max_number_of_raw_input < number_of_data_points:
        # only keep sensor data points from 1 to max_number_of_raw_input
        raw_sensor_data = raw_sensor_data.head(max_number_of_raw_input)
        # log the limitation
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(logging_level)
        logger.info("Limited %s data points from csv-File to %s data points", number_of_data_points,
                    max_number_of_raw_input)
    return raw_sensor_data


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

    z_helper.append_to_log_file(
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
                      filename_traces_basic,
                      csv_delimiter_traces_basic,
                      filename_parameters_file,
                      max_trace_length,
                      data_types,
                      data_types_list,
                      number_of_motion_sensors,
                      logging_level):
    """
    Split up the raw traces to shorter traces. The traces are limited by max_trace_length.

    :param traces_raw_pd: traces that will get divided
    :param data_sources_path: path of sources
    :param dir_runtime_files: the folder of the current run
    :param filename_traces_basic: filename of divided traces
    :param csv_delimiter_traces_basic: csv delimiter of divided trace file
    :param filename_parameters_file: filename of parameters file
    :param max_trace_length: the max length of one trace
    :param data_types: column/s which is/are applied to divided traces
    :param data_types_list: collection of all available data types
    :param number_of_motion_sensors: number of motion sensors
    :param logging_level: level of logging
    :return: The divided traces (final_vector), raw traces clustered into cases (output_case_traces_cluster), activated
    sensors grouped by divided traces (list_of_final_vectors_activations)
    """

    # start timer
    t0_divide_raw_traces = timeit.default_timer()
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    # ToDo ordentlicher machen
    list_of_final_vectors = []
    list_of_final_vectors_quantity = []
    list_of_activations = []
    list_of_final_vectors_activations = []
    list_header = [list(range(number_of_motion_sensors))]
    list_header[0].insert(0, 'Case')
    length_count = 1
    # creates new arrays with zeros in it
    # first index is reserved for divided_trace_case_id
    final_vector_time = np.zeros(number_of_motion_sensors + 1)
    final_vector_quantity = np.zeros(number_of_motion_sensors + 1, dtype=int)
    # current case looked at in raw traces (starts at first case which is the first row in the raw traces df)
    current_case_id = traces_raw_pd.iloc[0].Case
    # every divided trace will have its own id starting at 1
    divided_trace_case_id = 1
    # case id collection of all divided traces
    case_id_for_short_traces = []

    # runs through all raw traces and divides/assigns them to divided traces
    for data_row in traces_raw_pd.itertuples():

        # every divided trace only represents one raw case and has a length limitation
        if current_case_id != data_row.Case or length_count > max_trace_length:
            final_vector_time[0] = divided_trace_case_id
            final_vector_quantity[0] = divided_trace_case_id
            # Add data of last trace to final vectors
            list_of_final_vectors.append(final_vector_time)
            list_of_final_vectors_quantity.append(final_vector_quantity)
            list_of_final_vectors_activations.append(np.array(list_of_activations))
            # preparation for next trace
            current_case_id = data_row.Case
            divided_trace_case_id += 1
            length_count = 1
            final_vector_time = np.zeros(number_of_motion_sensors + 1)
            final_vector_quantity = np.zeros(number_of_motion_sensors + 1, dtype=int)
            list_of_activations = []
        # Set case to next case in list
        case_id_for_short_traces.append(divided_trace_case_id)
        list_of_activations.append(data_row.Sensor_Added)
        try:
            # add +1 to vector that counts quantity
            final_vector_quantity[data_row.Activity[-1] + 1] += 1
        except IndexError:
            # if no sensor activated, add quantity to M0
            final_vector_quantity[1] += 1
        # if no entry, M0 is active
        if len(data_row.Activity) == 0:
            # if row has a duration its added to the vector time of M0
            if not math.isnan(data_row.Duration):
                final_vector_time[1] += data_row.Duration

        for sensor_id in data_row.Activity:
            final_vector_time[int(sensor_id) + 1] += data_row.Duration
        length_count += 1

    # take last unfinished trace and copy it too
    final_vector_time[0] = divided_trace_case_id
    final_vector_quantity[0] = divided_trace_case_id
    list_of_final_vectors.append(final_vector_time)
    list_of_final_vectors_quantity.append(final_vector_quantity)
    list_of_final_vectors_activations.append(np.array(list_of_activations))

    final_vector_time = pd.DataFrame(list_of_final_vectors, columns=list_header[0])
    final_vector_quantity = pd.DataFrame(list_of_final_vectors_quantity, columns=list_header[0])
    final_vector_time.Case = final_vector_time.Case.astype(int)

    # apply short trace case ids to raw traces data
    output_case_traces_cluster = traces_raw_pd
    output_case_traces_cluster['Person'] = output_case_traces_cluster['Case']
    output_case_traces_cluster['Case'] = case_id_for_short_traces

    # which data types should the final vector contain?
    if data_types == 'quantity':
        # final vector contains how often a sensor was activated (in each trace)
        final_vector = final_vector_quantity
    elif data_types == 'time':
        # final vector contains how long a sensor was activated (in each trace)
        final_vector = final_vector_time
    elif data_types == 'quantity_time':
        # final vector contains both
        # drop caseID of second data frame
        final_vector_quantity = final_vector_quantity[final_vector_quantity.columns[1:]]
        final_vector = pd.concat([final_vector_time, final_vector_quantity],
                                 axis=1,
                                 ignore_index=True)
    else:
        # if invalid data types are selected the final vector can't get created.
        exception_msg = str(
            '\'' + data_types + '\' is not a valid choice for data types. Please choose one of the following: ' + str(
                data_types_list))
        logger.exception(exception_msg)
        raise ValueError(exception_msg)

    # stop timer
    t1_divide_raw_traces = timeit.default_timer()
    # calculate runtime
    runtime_divide_raw_traces = np.round(t1_divide_raw_traces - t0_divide_raw_traces, 1)

    z_helper.append_to_log_file(
        new_entry_to_log_variable='case_id_for_short_traces',
        new_entry_to_log_value=divided_trace_case_id - 1,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Number of shorter traces of length ' + str(max_trace_length) + '.')

    z_helper.append_to_log_file(
        new_entry_to_log_variable='runtime_' + inspect.stack()[0][3],
        new_entry_to_log_value=runtime_divide_raw_traces,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Seconds it took to divide long traces into shorter traces.')

    # logging
    logger.info("Dividing the long traces into %s short traces took %s seconds.",
                divided_trace_case_id - 1, runtime_divide_raw_traces)

    # write traces to disk
    final_vector.to_csv(data_sources_path + dir_runtime_files + filename_traces_basic, sep=csv_delimiter_traces_basic)
    # logging
    logger.info("Divided traces were saved as csv file '../%s",
                data_sources_path + dir_runtime_files + filename_traces_basic)

    return final_vector, output_case_traces_cluster, list_of_final_vectors_activations


def convert_raw_data_to_traces(raw_sensor_data,
                               dict_distance_adjacency_sensor,
                               data_sources_path,
                               dir_runtime_files,
                               filename_parameters_file,
                               filename_traces_raw,
                               csv_delimiter_traces,
                               logging_level,
                               number_of_motion_sensors,
                               prefix_motion_sensor_id='M',
                               distance_threshold=1.5,
                               max_number_of_people_in_house=2,
                               traces_time_out_threshold=300):
    """
    Creates traces out of the raw sensor data.

    :param raw_sensor_data: pandas data frame of sensor points
    :param dict_distance_adjacency_sensor: dictionary which contains a matrix which determine which sensors are near to each other
    :param data_sources_path: path of sources
    :param dir_runtime_files: directory of current run
    :param filename_parameters_file: filename of parameters file
    :param filename_traces_raw: filename of trace file which will get created
    :param csv_delimiter_traces: csv delimiter of trace file which will get created
    :param logging_level: level of logging
    :param number_of_motion_sensors: number of motion sensors
    :param prefix_motion_sensor_id: prefix of Motion-Sensor IDs
    :param distance_threshold: threshold when sensors are considered too far away
    :param max_number_of_people_in_house: maximum number of persons in house
    :param traces_time_out_threshold: the time in seconds in which a sensor activation is assigned to a existing trace
    :return: A pandas data frame containing the created traces.
    """

    # start timer
    t0_convert_raw2trace = timeit.default_timer()

    # get distance matrix from dictionary
    distance_matrix = dict_distance_adjacency_sensor['distance_matrix']

    # define columns of traces data frame
    pd_df_all_traces = pd.DataFrame(columns=['Activity', 'Sensor_Added', 'Duration', 'Timestamp', 'LC_Activity', 'LC'])

    # currently filled traces (not yet closed)
    trace_open = {}
    unique_new_trace_id = 1

    for data_row in raw_sensor_data.itertuples():
        # if not a motion sensor, continue with next row
        # ToDo: Maybe also accept door sensors for main entries
        if data_row.SensorID[0:len(prefix_motion_sensor_id)] != prefix_motion_sensor_id:
            continue

        # extract sensor ID and skip prefix:
        # ToDo: Do this for all elements of that column under the condition of it being an M Sensor
        sensor_id_no_prefix = int(data_row.SensorID[len(prefix_motion_sensor_id):])

        # if sensor is greater than the highest sensor id, skip and report
        if sensor_id_no_prefix > number_of_motion_sensors - 1:
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
            # pd_df_all_traces_short.loc[trace_to_close] = [trace_to_close, added_sensors_list]
            # remove Sensor_Added Column
            # temporary_df.drop('Sensor_Added', axis=1, inplace=True)
            # todo: Move to end and do them all at once to save time
            pd_df_all_traces = pd.concat([pd_df_all_traces, temporary_df], axis=0, join='outer', ignore_index=False,
                                         keys=None, levels=None, names=None, verify_integrity=False, copy=True,
                                         sort=False)
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
    # adjust traces data frame
    # replace 'nan' with 0
    pd_df_all_traces['Sensor_Added'].fillna(0, inplace=True)
    # Convert Float to integer format
    pd_df_all_traces['Sensor_Added'] = pd.to_numeric(pd_df_all_traces['Sensor_Added'], downcast='signed')
    pd_df_all_traces['LC_Activity'] = pd.to_numeric(pd_df_all_traces['LC_Activity'], downcast='signed')
    # sorting
    pd_df_all_traces = pd_df_all_traces.sort_values(by=['Case', 'Timestamp'])

    # reset index so it matches up after the sorting
    pd_df_all_traces = pd_df_all_traces.reset_index(drop=True)

    # stop timer
    t1_convert_raw2trace = timeit.default_timer()
    # calculate method runtime
    runtime_convert_raw2trace = np.round(t1_convert_raw2trace - t0_convert_raw2trace, 1)
    # logging
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Extracting %s traces took %s seconds.", unique_new_trace_id - 1, runtime_convert_raw2trace)

    # write traces to disk
    utils.write_csv_file(data=pd_df_all_traces, filedir=data_sources_path + dir_runtime_files,
                         filename=filename_traces_raw, separator=csv_delimiter_traces, logging_level=logging_level)
    # logging
    logger.info("Traces were saved as csv file '../%s", data_sources_path + dir_runtime_files + filename_traces_raw)

    return pd_df_all_traces


def read_in_sensor_data(data_sources_path, filename_sensor_data, rel_dir_name_sensor_data, csv_delimiter_sensor_data,
                        csv_header_sensor_data, csv_parse_dates_sensor_data, csv_dtype_sensor_data, logging_level):
    """
    Reads the sensor data file and returns the content of the file as pandas data frame.

    :param data_sources_path: Path where the source files can be found
    :param filename_sensor_data: Name of the file
    :param rel_dir_name_sensor_data:  Name of the folder where the sensor data is located
    :param csv_delimiter_sensor_data: Char that separates columns in csv files
    :param csv_header_sensor_data: Row number/s to use as the column names, and the start of the data.
    :param csv_parse_dates_sensor_data: columns that get passed as dates.
    :param csv_dtype_sensor_data: data types of the columns
    :param logging_level: level of logging
    :return: Pandas data frame containing the sensor information of file in given folder
    """

    # start timer
    t0_read_csv_files = timeit.default_timer()

    # creates pandas data frame out of input of a csv file
    data_frame = utils.read_csv_file(filedir=data_sources_path,
                                     filename=filename_sensor_data, separator=csv_delimiter_sensor_data,
                                     header=csv_header_sensor_data,
                                     parse_dates=csv_parse_dates_sensor_data, dtype=csv_dtype_sensor_data,
                                     logging_level=logging_level)

    # calculate how many data points there are
    number_of_data_points = data_frame.shape[0]

    # stop timer
    t1_read_csv_files = timeit.default_timer()
    # calculate runtime
    runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Extracted %s data points from csv-File on disc in %s seconds",
                number_of_data_points, runtime_read_csv_files)

    return data_frame