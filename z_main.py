from sklearn.model_selection import ParameterGrid
# supports logging
import logging

from datetime import datetime
import timeit
import numpy as np
import pandas as pd
import os

# import settings file
import z_setting_parameters as settings
import z_create_distance_matrix as create_dm
import z_helper
import a_EventCaseCorrelation as ecc
import b_ActivityDiscovery as ad

# start timer
t0_main = timeit.default_timer()

# get distance Matrix from imported adjacency-matrix
dict_distance_adjacency_sensor = create_dm.get_distance_matrix()

# draw a node-representation of the Smart Home
create_dm.draw_adjacency_graph(dict_room_information=dict_distance_adjacency_sensor,
                               data_sources_path=settings.path_data_sources,
                               dir_runtime_files=settings.dir_runtime_files,
                               filename_adjacency_plot=settings.filename_adjacency_plot)

param_grid = {'zero_distance_value': settings.zero_distance_value_list,
              'distance_threshold': settings.distance_threshold_list,
              'traces_time_out_threshold': settings.traces_time_out_threshold_list,
              'max_trace_length': settings.max_trace_length_list,
              'k_means_number_of_clusters': settings.k_means_number_of_clusters}

grid = ParameterGrid(param_grid)

# count number of iteration so code can be continued after interruption
iteration_counter = 0

# pandas data frame which contains
df_all_data = z_helper.read_csv_files()

# runs throw all combinations of the parameters (zero_distance_value, distance_threshold, ...)
# https://stackoverflow.com/questions/13370570/elegant-grid-search-in-python-numpy
for params in grid:
    iteration_counter += 1
    # folder name containing files read and written during runtime
    dir_runtime_files = 'runtime-files' + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # create new folder for current run
    if not os.path.exists(settings.path_data_sources + dir_runtime_files):
        os.makedirs(settings.path_data_sources + dir_runtime_files)

    z_helper.create_parameters_log_file(dir_runtime_files=dir_runtime_files,
                                        grid_search_parameters=params)
    # Logger configuration
    logging.basicConfig(
        level=settings.logging_level,
        format="%(asctime)s [%(levelname)-5.5s] [%(threadName)-12.12s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[logging.FileHandler("{0}/{1}.log".format(settings.path_data_sources + dir_runtime_files,
                                                           settings.filename_log_file)), logging.StreamHandler()])
    logger = logging.getLogger('main')
    logger.setLevel(settings.logging_level)

    dict_distance_adjacency_sensor['distance_matrix'] = \
        create_dm.set_zero_distance_value(distance_matrix=dict_distance_adjacency_sensor['distance_matrix'],
                                          zero_distance_value=params['zero_distance_value'])

    #################### EventCaseCorrelation ####################
    # transform raw-data to traces
    trace_data_time, output_case_traces_cluster, list_of_final_vectors_activations = \
        ecc.create_trace_from_file(data_sources_path=settings.path_data_sources,
                                   dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                   dir_name_sensor_data=settings.dir_name_sensor_data,
                                   csv_delimiter=settings.csv_delimiter,
                                   filename_parameters_file=settings.filename_parameters_file,
                                   prefix_motion_sensor_id=settings.prefix_motion_sensor_id,
                                   dir_runtime_files=dir_runtime_files,
                                   distance_threshold=settings.distance_threshold,
                                   max_number_of_people_in_house=settings.max_number_of_people_in_house,
                                   traces_time_out_threshold=settings.traces_time_out_threshold,
                                   data_types=settings.data_types,
                                   max_trace_length=settings.max_trace_length,
                                   max_number_of_raw_input=settings.max_number_of_raw_input)

    # cut away the case number for SOM training
    trace_data_without_case_number = trace_data_time[trace_data_time.columns[1:]]

    #################### ActivityDiscovery ####################
    # ### k-Means ###
    k_means_cluster_id = ad.custom_kmeans(data=trace_data_without_case_number,
                                       number_of_clusters=settings.K_opt)

    sm, km, quantization_error, topographic_error = ad.merge_related_tracking_entries(
        trace_data_without_case_number=trace_data_without_case_number,
        K_opt=settings.K_opt, path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file, loggers=logger)

    # write BMU to output file
    trace_data_time['BMU'] = np.transpose(sm._bmu[0, :]).astype(int)

    # write k-Means Cluster to output file
    cluster_list = km.labels_[trace_data_time.BMU]
    trace_data_time['Cluster'] = pd.DataFrame(cluster_list)

    # write vanilla kmeans to output file
    trace_data_time['kMeansVanilla'] = np.transpose(k_means_cluster_id).astype(int)

    # write BMU and Cluster to output_case file
    # so the raw_data from the beginning now also has clusters
    output_case_traces_cluster['Cluster'] = trace_data_time['Cluster'][
        output_case_traces_cluster['Case'] - 1].values
    output_case_traces_cluster['BMU'] = trace_data_time['BMU'][
        output_case_traces_cluster['Case'] - 1].values
    output_case_traces_cluster['kMeansVanilla'] = k_means_cluster_id[
        output_case_traces_cluster['Case'] - 1]

    trace_data_time = trace_data_time.sort_values(by=['Cluster', 'BMU'])
    trace_data_time = trace_data_time.reset_index(drop=True)

    # write traces to disk
    trace_data_time.to_csv(settings.path_data_sources + dir_runtime_files + '/Cluster.csv',
                           sep=';',
                           index=None)

    output_case_traces_cluster.to_csv(
        settings.path_data_sources + dir_runtime_files + '/Cases_Cluster.csv',
        sep=';',
        index=None)

    # stop timer
    t1_main = timeit.default_timer()

    # calculate runtime
    runtime_main = np.round(t1_main - t0_main, 1)
    z_helper.append_to_performance_documentation_file(
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        list_of_properties={'quantization_error': quantization_error,
                            'topographic_error': topographic_error,
                            'runtime_main': runtime_main,
                            'zero_distance_value': settings.zero_distance_value,
                            'distance_threshold': settings.distance_threshold,
                            'max_number_of_people_in_house': settings.max_number_of_people_in_house,
                            'traces_time_out_threshold': settings.traces_time_out_threshold,
                            'max_trace_length': settings.max_trace_length,
                            'data_types': settings.data_types,
                            'k_means_number_of_clusters': settings.K_opt,
                            'vector_size_word2vec': settings.vector_size})

    z_helper.append_to_log_file(
        new_entry_to_log_variable='runtime_main',
        new_entry_to_log_value=runtime_main,
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        new_entry_to_log_description='Total runtime in seconds.')

    z_helper.append_to_log_file(
        new_entry_to_log_variable='execution_completed',
        new_entry_to_log_value=True,
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        new_entry_to_log_description='Successfully executed code.')
    logger.info("Total runtime: %s", runtime_main)
    print("Done with iteration: ", iteration_counter,
          " ###################################")
pass
