print("hallo welt")
from sklearn.model_selection import ParameterGrid
# supports logging
import logging

from datetime import datetime
import timeit

import os
# import settings file
import z_setting_parameters as settings
import z_create_distance_matrix as create_dm
import z_helper

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

df_all_data = z_helper.read_csv_files()

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

    print(2)
    # your_function(params['param1'], params['param2'])
    pass
