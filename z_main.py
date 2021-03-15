import inspect
import logging

from datetime import datetime
import timeit
import numpy as np
from hyperopt import hp, fmin, Trials, STATUS_OK
import os
import pathlib

# import settings file
from u_utils import u_helper as helper, u_DistanceMatrixCreation as create_dm
from u_utils import u_utils
import z_setting_parameters as settings
from a_EventCaseCorrelation import a_EventCaseCorrelation as ecc
from b_ActivityDiscovery import b_ActivityDiscovery as ad
from c_EventActivityAbstraction import c_EventActivityAbstraction as eaa
from d_ProcessDiscovery import d_ProcessDiscovery as prd

# start timer
t0_main = timeit.default_timer()

# check if runtime folder exists, if not create it
path = pathlib.Path(settings.path_data_sources + settings.dir_runtime_files)
path.mkdir(parents=True, exist_ok=True)

# Logger configuration
logging.basicConfig(
    level=settings.logging_level,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(settings.path_data_sources + settings.dir_runtime_files + settings.filename_log_file),
        logging.StreamHandler()])
# "disable" specific logger
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('hyperopt').setLevel(logging.WARNING)

# checks settings for correctness
helper.check_settings(zero_distance_value_min=settings.zero_distance_value_min,
                      zero_distance_value_max=settings.zero_distance_value_max,
                      distance_threshold_min=settings.distance_threshold_min,
                      distance_threshold_max=settings.distance_threshold_max,
                      traces_time_out_threshold_min=settings.traces_time_out_threshold_min,
                      traces_time_out_threshold_max=settings.traces_time_out_threshold_max,
                      trace_length_limit_min=settings.trace_length_limit_min,
                      trace_length_limit_max=settings.trace_length_limit_max,
                      k_means_number_of_clusters_min=settings.k_means_number_of_clusters_min,
                      k_means_number_of_clusters_max=settings.k_means_number_of_clusters_max,
                      event_case_correlation_method=settings.event_case_correlation_method,
                      event_case_correlation_method_list=settings.event_case_correlation_method_list,
                      data_types=settings.data_types,
                      data_types_list=settings.data_types_list,
                      miner_type=settings.miner_type,
                      miner_type_list=settings.miner_type_list,
                      logging_level=settings.logging_level)

# get distance Matrix from imported adjacency-matrix
dict_distance_adjacency_sensor = create_dm.get_distance_matrix()

# draw a node-representation of the Smart Home
create_dm.draw_adjacency_graph(dict_room_information=dict_distance_adjacency_sensor,
                               data_sources_path=settings.path_data_sources,
                               filename_adjacency_plot=settings.filename_adjacency_plot)

# load data
# read in the sensor data as a pandas data frame
raw_sensor_data = u_utils.read_csv_file(filedir=settings.path_data_sources,
                                        filename=settings.filename_sensor_data,
                                        separator=settings.csv_delimiter_sensor_data,
                                        header=settings.csv_header_sensor_data,
                                        parse_dates=settings.csv_parse_dates_sensor_data,
                                        dtype=settings.csv_dtype_sensor_data,
                                        )


def perform_process_model_discovery(params):
    # count number of iterations
    perform_process_model_discovery.iteration_counter += 1

    # apply current time to the format of the folder name containing files read and written during runtime
    dir_runtime_files = datetime.now().strftime(settings.dir_runtime_files + settings.dir_runtime_files_iteration)

    # create new folder for current run
    if not os.path.exists(settings.path_data_sources + dir_runtime_files):
        os.makedirs(settings.path_data_sources + dir_runtime_files)

    helper.create_parameters_log_file(dir_runtime_files=dir_runtime_files, params=params)

    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)
    logger.info("################# Start iteration %s of %s #################",
                perform_process_model_discovery.iteration_counter, settings.opt_attempts)

    dict_distance_adjacency_sensor['distance_matrix'] = \
        create_dm.set_zero_distance_value(distance_matrix=dict_distance_adjacency_sensor['distance_matrix'],
                                          zero_distance_value=params['zero_distance_value'])

    # ################### EventCaseCorrelation ####################
    # transform raw-data to traces
    trace_data_time, output_case_traces_cluster, list_of_final_vectors_activations = \
        ecc.choose_event_case_correlation_method(method=settings.event_case_correlation_method,
                                                 dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                                 dir_runtime_files=dir_runtime_files,
                                                 distance_threshold=params['distance_threshold'],
                                                 traces_time_out_threshold=params['traces_time_out_threshold'],
                                                 trace_length_limit=params['trace_length_limit'],
                                                 raw_sensor_data=raw_sensor_data,
                                                 max_errors_per_day=params['max_errors_per_day'],
                                                 logging_level=settings.logging_level)

    # cut away the case number for SOM training
    trace_data_without_case_number = trace_data_time[trace_data_time.columns[1:]]

    # ################### ActivityDiscovery ####################
    # k-means clustering and SOM classification
    k_means_cluster_ids, sm, km, quantization_error, topographic_error = ad.cluster_and_classify_activities(
        trace_data_without_case_number=trace_data_without_case_number,
        number_of_clusters=params['k_means_number_of_clusters'], K_opt=params['k_means_number_of_clusters'],
        path_data_sources=settings.path_data_sources, dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        number_of_motion_sensors=settings.number_of_motion_sensors, logging_level=settings.logging_level)

    # ################### EventActivityAbstraction ####################
    output_case_traces_cluster = \
        eaa.create_event_log_files(trace_data_time=trace_data_time,
                                   output_case_traces_cluster=output_case_traces_cluster,
                                   k_means_cluster_ids=k_means_cluster_ids,
                                   path_data_sources=settings.path_data_sources,
                                   dir_runtime_files=dir_runtime_files,
                                   sm=sm, km=km, filename_cluster=settings.filename_cluster,
                                   csv_delimiter_cluster=settings.csv_delimiter_cluster,
                                   filename_cases_cluster=settings.filename_cases_cluster,
                                   csv_delimiter_cases_cluster=settings.csv_delimiter_cases_cluster)

    # ################### ProcessDiscovery ####################
    prd.create_activtiy_models(output_case_traces_cluster=output_case_traces_cluster,
                               path_data_sources=settings.path_data_sources, dir_runtime_files=dir_runtime_files,
                               dir_dfg_cluster_files=settings.dir_dfg_files,
                               filename_dfg_cluster=settings.filename_dfg_cluster,
                               rel_proportion_dfg_threshold=settings.rel_proportion_dfg_threshold,
                               logging_level=settings.logging_level)

    metrics = prd.create_process_model(output_case_traces_cluster=output_case_traces_cluster,
                                       path_data_sources=settings.path_data_sources,
                                       dir_runtime_files=dir_runtime_files,
                                       filename_log_export=settings.filename_log_export,
                                       dir_petri_net_files=settings.dir_petri_net_files,
                                       filename_petri_net=settings.filename_petri_net,
                                       dir_dfg_files=settings.dir_dfg_files,
                                       filename_dfg=settings.filename_dfg,
                                       rel_proportion_dfg_threshold=settings.rel_proportion_dfg_threshold,
                                       miner_type=settings.miner_type,
                                       logging_level=settings.logging_level)

    # stop timer
    t1_main = timeit.default_timer()

    # calculate runtime
    runtime_main = np.round(t1_main - t0_main, 1)
    helper.append_to_performance_documentation_file(
        path_data_sources=settings.path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_benchmark=settings.filename_benchmark,
        csv_delimiter_benchmark=settings.csv_delimiter_benchmark,
        list_of_properties={'quantization_error': quantization_error,
                            'topographic_error': topographic_error,
                            'runtime_main': runtime_main,
                            'zero_distance_value': params['zero_distance_value'],
                            'distance_threshold': params['distance_threshold'],
                            'max_number_of_people_in_house': settings.max_number_of_people_in_house,
                            'traces_time_out_threshold': params['traces_time_out_threshold'],
                            'trace_length_limit': params['trace_length_limit'],
                            'data_types': settings.data_types,
                            'k_means_number_of_clusters': params['k_means_number_of_clusters']})

    helper.append_to_log_file(
        new_entry_to_log_variable='runtime_main',
        new_entry_to_log_value=runtime_main,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        new_entry_to_log_description='Total runtime in seconds.')

    helper.append_to_log_file(
        new_entry_to_log_variable='execution_completed',
        new_entry_to_log_value=True,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=settings.filename_parameters_file,
        new_entry_to_log_description='Successfully executed code.')

    logger.info("Total runtime: %s", runtime_main)
    logger.info("################# End iteration %s of %s #################",
                perform_process_model_discovery.iteration_counter, settings.opt_attempts)

    return {
        # ToDo: Is the current loss value the right one?
        'loss': -metrics['fitness']['log_fitness'],
        'status': STATUS_OK,
        'iteration': perform_process_model_discovery.iteration_counter,
        'dir_runtime_files': dir_runtime_files,
        'opt_params': params
    }


# creates a selection of thresholds out of min, max and threshold length
distance_threshold_list = helper.create_distance_threshold_list(
    distance_threshold_min=settings.distance_threshold_min,
    distance_threshold_max=settings.distance_threshold_max,
    distance_threshold_step_length=settings.distance_threshold_step_length)

# hyperopt parameter tuning
# parameter's search space
space = {
    'zero_distance_value': hp.randint('zero_distance_value', settings.zero_distance_value_min,
                                      settings.zero_distance_value_max + 1),
    'traces_time_out_threshold': hp.randint('traces_time_out_threshold', settings.traces_time_out_threshold_min,
                                            settings.traces_time_out_threshold_max + 1),
    'trace_length_limit': hp.randint('trace_length_limit', settings.trace_length_limit_min,
                                     settings.trace_length_limit_max + 1),
    'k_means_number_of_clusters': hp.randint('k_means_number_of_clusters', settings.k_means_number_of_clusters_min,
                                             settings.k_means_number_of_clusters_max + 1),
    'distance_threshold': hp.choice('distance_threshold', distance_threshold_list),
    'max_errors_per_day': hp.randint('max_errors_per_day', settings.max_errors_per_day_min,
                                     settings.max_errors_per_day_max + 1)
}

# capture the iterations of hyperopt parameter tuning
perform_process_model_discovery.iteration_counter = 0
trials = Trials()
# perform process model discovery for different parameter combinations and find the best outcome (hyperopt parameter tuning)
fmin(fn=perform_process_model_discovery,
     space=space,
     algo=settings.opt_algorithm,
     max_evals=settings.opt_attempts,
     verbose=False,
     trials=trials)

# additional information of the best iteration
best = trials.best_trial
information_string = '\nbest iteration:\n\toptimised function value = ' + \
                     str(-best['result']['loss']) + '\n\toptimised parameters:'
for key, value in best['result']['opt_params'].items():
    information_string += '\n\t\t' + str(key) + ' = ' + str(value)
information_string += '\n\tfiles directory = ' + str(best['result']['dir_runtime_files']) + '\n\titeration = ' + str(
    best['result']['iteration'])

# logger to print and save the information
logger = logging.getLogger('main')
logger.setLevel(settings.logging_level)
logger.info(information_string)
pass
