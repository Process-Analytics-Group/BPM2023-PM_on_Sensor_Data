"""
Code from Master Project
"""
import a_EventCaseCorrelation.FreFraLa.Filter as FreFraLa_Filter


def apply_threshold_filtering(dict_distance_adjacency_sensor,
                              max_errors_per_day,
                              traces_time_out_threshold,
                              trace_length_limit,
                              raw_sensor_data):
    # filter input dataset for errors and time
    filtered_dataset = \
        FreFraLa_Filter.filter_visitor_days(threshold=max_errors_per_day,
                                            raw_sensor_data=raw_sensor_data,
                                            dict_distance_adjacency_sensor=dict_distance_adjacency_sensor)

    # transform raw data into vectors

