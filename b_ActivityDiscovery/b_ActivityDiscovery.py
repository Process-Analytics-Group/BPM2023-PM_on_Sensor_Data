# SOM
import inspect
import logging
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn_som.som import SOM
from yellowbrick.cluster import KElbowVisualizer
from hyperopt import fmin
from functools import partial

import b_ActivityDiscovery.FreFlaLa.b_FreFlaLa as b_FreFlaLa
import z_setting_parameters as settings
from b_ActivityDiscovery.self_organizing_map.sompy import SOMFactory
from u_utils import u_helper as helper, u_utils as utils


def choose_and_perform_clustering_method(clustering_method,
                                         number_of_clusters,
                                         trace_data_without_case_number,
                                         dir_runtime_files,
                                         dict_distance_adjacency_sensor,
                                         vectorization_type,
                                         min_number_of_clusters,
                                         max_number_of_clusters):
    """
    This method manages the different clustering methods and starts the selected method. The results of the different
    clustering methods is in the same Format.

    @param clustering_method:               Specifying the clustering method that is used
    @param number_of_clusters:              Specifying the number of clusters. (not used for "k-Means-Elbow" and
                                            "k-Medoids-Elbow")
    @param trace_data_without_case_number:  List of all vectors that should be clustered
    @param dir_runtime_files:
    @param logging_level:
    @param dict_distance_adjacency_sensor:  Dictionary with distance- and adjacency-matrix
    @param vectorization_type:              Specifying if the values in the dataset are the number of occurrences of the
                                            sensor or the time a sensor was activated
    @param min_number_of_clusters:          minimum number of clusters for the elbow-method (only used in
                                            "k-Means-Elbow" and "k-Medoids-Elbow")
    @param max_number_of_clusters:          maximum number of clusters for the elbow-method (only used in
                                            "k-Means-Elbow" and "k-Medoids-Elbow")

    @return:                                result list returns cluster for each vector
    """

    # clustering with self organizing map
    if clustering_method == 'SOMPY':
        k_means_cluster_ids, sm, km = cluster_and_classify_activities(
            trace_data_without_case_number=trace_data_without_case_number,
            max_number_of_clusters=max_number_of_clusters, min_number_of_clusters=min_number_of_clusters,
            dir_runtime_files=dir_runtime_files)
        # use the k-means inertia as clustering score. Average distance to centroids
        cluster_score = km.inertia_
        clustering_result = km.labels_[np.transpose(sm._bmu[0, :]).astype(int)]

    # clustering with the self organizing map by sklearn
    elif clustering_method == 'sklearn-SOM':
        # find all possible som dimensions
        divisor_pairs = utils.find_divisor_pairs(number=number_of_clusters)
        # only keep the most "squared" shape
        som_dimensions = divisor_pairs[int((divisor_pairs.__len__() - 1) / 2)]

        # perform process model discovery for different parameter combinations and find the best outcome
        space = helper.create_som_param_opt_space()
        # helper variables
        choose_and_perform_clustering_method.predictions = []
        choose_and_perform_clustering_method.inertia = sys.maxsize
        # find the best matching SOM (hyperparameter optimization) with fixed dimensions
        fmin(fn=partial(create_som_with_sklearn, m=som_dimensions[0], n=som_dimensions[1],
                        trace_data_without_case_number=trace_data_without_case_number),
             space=space,
             algo=settings.som_opt_algorithm,
             max_evals=settings.som_opt_attempts,
             verbose=False)
        clustering_result = choose_and_perform_clustering_method.predictions

        # logger
        number_of_clusters = max(clustering_result) + 1
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(settings.logging_level)
        logger.info("Clustered data into %s clusters using a %sx%s sklearn SOM.", number_of_clusters, som_dimensions[0],
                    som_dimensions[1])


    # clustering with a custom distance calculation
    elif clustering_method == 'CustomDistance':
        clustering_result = b_FreFlaLa.clustering_with_custom_distance_calculation(
            trace_data_without_case_number,
            dict_distance_adjacency_sensor,
            vectorization_type,
            number_of_clusters)

    # clustering with k-means form sk-learn
    elif clustering_method == 'k-Means':

        clustering_result, cluster_score = b_FreFlaLa.clustering_kmeans(trace_data_without_case_number,
                                                                        number_of_clusters)

    # elbow method to choose optimal number of clusters and clustering with k-means
    elif clustering_method == 'k-Means-Elbow':
        # if the maximum und minimum number of clusters are equal the elobw-method is skipped
        if max_number_of_clusters == min_number_of_clusters:
            clustering_result, cluster_score = b_FreFlaLa.clustering_kmeans(trace_data_without_case_number,
                                                                            max_number_of_clusters)
        else:
            clustering_result, cluster_score = b_FreFlaLa.elbow_method_kmeans(trace_data_without_case_number,
                                                                              min_number_of_clusters,
                                                                              max_number_of_clusters)

    # clustering with k-medoids form sk-learn-extra
    elif clustering_method == 'k-Medoids':

        clustering_result, cluster_score = b_FreFlaLa.clustering_k_medoids(trace_data_without_case_number,
                                                                           number_of_clusters)

    # elbow method to choose optimal number of clusters and clustering with k-medoids
    elif clustering_method == 'k-Medoids-Elbow':

        # if the maximum und minimum number of clusters are equal the elobw-method is skipped
        if max_number_of_clusters == min_number_of_clusters:
            clustering_result, cluster_score = b_FreFlaLa.clustering_k_medoids(trace_data_without_case_number,
                                                                               number_of_clusters)
        else:
            clustering_result, cluster_score = b_FreFlaLa.elbow_method_kmedoids(trace_data_without_case_number,
                                                                                min_number_of_clusters,
                                                                                max_number_of_clusters)

    else:
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(settings.logging_level)
        error_msg = "'" + clustering_method + "' is not a valid clustering method. Please check the settings."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return clustering_result


def create_som_with_sklearn(params, m, n, trace_data_without_case_number):
    """
    Creates a self organizing map for given dimensions, parameters and data.
    :param params: hyperopt parameters to find the best SOM for given data
    :param m: vertical dimension of the som
    :param n: horizontal dimension of the som
    :param trace_data_without_case_number: the data the SOM is build on
    :return: the inertia of the current som
    """
    # creates a SOM
    sklearn_som = SOM(m=m, n=n, lr=params['lr'], sigma=params['sigma'], dim=trace_data_without_case_number.shape[1])
    # "adapts" the SOM to the data
    sklearn_som.fit(trace_data_without_case_number.values)
    # find the best performing SOM
    if sklearn_som.inertia_ < choose_and_perform_clustering_method.inertia:
        choose_and_perform_clustering_method.predictions = sklearn_som.predict(trace_data_without_case_number.values)
        choose_and_perform_clustering_method.inertia = sklearn_som.inertia_

    return sklearn_som.inertia_


def cluster_and_classify_activities(trace_data_without_case_number, min_number_of_clusters, max_number_of_clusters,
                                    dir_runtime_files):
    # Instantiate the clustering model and visualizer
    model = KMeans()
    elbow_kmeans = KElbowVisualizer(model, k=(min_number_of_clusters, max_number_of_clusters))

    # Fit the data to the elbow method
    elbow_kmeans.fit(trace_data_without_case_number)
    # Visualize the elbow method
    elbow_kmeans.show()
    # optimal number of clusters determined by the elbow method
    number_of_clusters = elbow_kmeans.elbow_value_
    # k-means clustering with number of clusters form elbow method
    k_means_cluster_ids = custom_kmeans(data=trace_data_without_case_number,
                                        number_of_clusters=number_of_clusters)

    # self organizing map with number of clusters form elbow method
    sm, km, quantization_error, topographic_error = self_organising_map(
        trace_data_without_case_number=trace_data_without_case_number, K_opt=number_of_clusters,
        dir_runtime_files=dir_runtime_files)

    return k_means_cluster_ids, sm, km


# k-means Vanilla
def custom_kmeans(data, number_of_clusters):
    np_data_array = data.values
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np_data_array)
    return kmeans.labels_


# SOM with k-means to cluster the best matching units
def self_organising_map(trace_data_without_case_number, K_opt, dir_runtime_files):
    path_data_sources = settings.path_data_sources
    filename_parameters_file = settings.filename_parameters_file
    number_of_motion_sensors = settings.number_of_motion_sensors
    # create list with sensor names
    names = []
    for sensor_number in range(0, number_of_motion_sensors - 1):
        names.append(str("Sensor " + str(sensor_number)))
    # create the sompy SOM network and train it
    sm = SOMFactory().build(trace_data_without_case_number.values, normalization='var',
                            initialization='pca',
                            component_names=names)
    sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

    # The quantization error: average distance between each data vector and its BMU.
    # The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
    topographic_error = sm.calculate_topographic_error()
    quantization_error = np.mean(sm._bmu[1])

    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)
    logger.info("Topographic error = %s; Quantization error = %s", topographic_error, quantization_error)

    K = 20  # stop at this k for SSE sweep
    [labels, km, norm_data] = sm.cluster(n_clusters=K, opt=K_opt)

    helper.append_to_log_file(
        new_entry_to_log_variable='topographic_error',
        new_entry_to_log_value=topographic_error,
        path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='The topographic error: the proportion of all data '
                                     'vectors for which first and second BMUs are not adjacent units.')
    helper.append_to_log_file(
        new_entry_to_log_variable='quantization_error',
        new_entry_to_log_value=quantization_error,
        path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='The quantization error: '
                                     'average distance between each data vector and its BMU.')

    helper.append_to_log_file(
        new_entry_to_log_variable='k_means_number_of_clusters',
        new_entry_to_log_value=K_opt,
        path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Number of clusters used to do the vanilla k-means '
                                     'and Cluster the SOM-Neurons. ')

    return sm, km, quantization_error, topographic_error
