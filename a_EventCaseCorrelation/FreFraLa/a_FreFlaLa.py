"""
Code from Master Project
"""
import pandas as pd
import a_EventCaseCorrelation.FreFraLa.Filter as FreFraLa_Filter
import a_EventCaseCorrelation.FreFraLa.vectoring as FreFraLa_Vectoring
import timeit
import numpy as np

def apply_threshold_filtering(dict_distance_adjacency_sensor,
                              max_errors_per_day,
                              traces_time_out_threshold,
                              trace_length_limit,
                              raw_sensor_data,
                              vectorization_method,
                              logging_level):
    # start timer
    t0_main = timeit.default_timer()

    # convert sensor ids to int for better efficiency, drop the 'M'
    # ToDo: Move Sensor type identifier in separate column in future versions
    raw_sensor_data_sensor_int = raw_sensor_data.copy(deep=True)
    raw_sensor_data_sensor_int['SensorID'] = raw_sensor_data_sensor_int['SensorID'].str.replace('M', '')
    raw_sensor_data_sensor_int['SensorID'] = pd.to_numeric(raw_sensor_data_sensor_int['SensorID'])

    # filter input dataset for errors and time
    filtered_dataset = \
        FreFraLa_Filter.filter_visitor_days(threshold=max_errors_per_day,
                                            raw_sensor_data_sensor_int=raw_sensor_data_sensor_int,
                                            dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                            logging_level=logging_level)

    # transform raw data into vectors
    FreFraLa_method = 'byRooms'
    case_vectors, raw_sensor_data_sensor_int = \
        FreFraLa_Vectoring.choose_method(filtered_dataset=filtered_dataset,
                                         method=FreFraLa_method,
                                         dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                         vectorization_method=vectorization_method)

    # stop timer
    t1_main = timeit.default_timer()

    # calculate runtime
    print('#######' + str(np.round(t1_main - t0_main, 1)))
    return case_vectors, raw_sensor_data_sensor_int
