import inspect
import logging

from scipy.cluster.hierarchy import fclusterdata
# ToDo @DJ: wieder einkommentieren
from sklearn_extra.cluster import KMedoids
import numpy as np
import pandas as pd


def clustering_k_medoids(allvectors, clustersize):
    '''
    K-Medoids using sklearn_extra

    @param allvectors:  list of all vectors that should be clustered.
    @param clustersize: Specifies the number of cluster. Default is 15.

    @return:            result list returns cluster for each vector
    '''

    return KMedoids(n_clusters=clustersize, random_state=0).fit(allvectors).labels_


def clustering_with_custom_distance_calculation(allvectors, dict_distance_adjacency_sensor, vectorization_type,
                                                clustersize=15, linkage_method_for_clustering='ward',
                                                logging_level=None):
    '''
    Clusters the dataset using custom calculation.

    @param allvectors:                      A list of all vectors that should be clustered.
    @param dict_distance_adjacency_sensor:  A Dictionary with a "distance_matrix" inside
    @param vectorization_type:              Specifies if the vector is only time or sensor quantity or if its both
    @param clustersize:                     Specifies the number of cluster. Default is 15.
    @param linkage_method_for_clustering:   Specifies the linkage method. Default is 'ward'. The linkage method to use
                                            (single, complete, average, weighted, median centroid, ward)

    @return:                                result list returns cluster for each vector
    '''

    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("start modified clustering method")

    # distance calculation
    def euclidean_and_most_used_sensor_dist(v1, v2):
        '''
        -- FIRST ENTRIES OF THE VECTORS IS ONLY INDEX AND WILL BE DELETED BEFORE CLUSTERING OR CALCULATION --

        distance calculation between two vectors that is used for clustering.

        1. The Euclidean distance between the Vectors is calculated.
        2. The most used Sensor of each vector are identified. Than the needed steps over
           neighbouring sensors, between this two sensors, are counted.
        3. Euclidean distance * needed steps = calculated distance between two vectors

        @param v1: first vector (first entry is only index for identification, has to be deleted before clustering or
            calculation)
        @param v2: second vector @return: calculated distance between two vectors (first entry is only index for
            identification, has to be deleted before clustering or calculation)
        '''

        # euclidean distance * Steps between most used sensors.
        # euclidean distance: first value is deleted because it is just for identifing the vector
        return np.linalg.norm(np.delete(v1, 0) - np.delete(v2, 0)) * \
               dict_distance_adjacency_sensor["distance_matrix"][indices_of_most_used_sensor_per_vector[v1[0]]][
                   indices_of_most_used_sensor_per_vector[v2[0]]]

    # decision between 'quantity_time' and 'quantity'/'time', because of different vector length
    if vectorization_type == 'quantity_time':

        # slicing the "Zero-sensor" and the quantity count off
        allvectors_short = allvectors.iloc[:, ((len(allvectors.columns) // 2) + 1):]

    else:
        # slicing the "Zero-sensor" off
        allvectors_short = allvectors.iloc[:, 1:]

    # replace sensornames with index number
    allvectors_short.columns = range(1, len(allvectors_short.columns) + 1)

    # find the most used sensor in every vector and create a Series with the corresponding sensor number
    indices_of_most_used_sensor_per_vector = allvectors_short.idxmax(axis=1)

    # adding the vector-index as first element in vector to identify the vector in the custom distace calculation.
    # This value is removed before clustering
    allvectors.insert(loc=0, column='index', value=(range(1, len(allvectors.index) + 1)))

    # clustering method that can use a custom distance calculation. 'maxclust' and t=15 -> 15 cluster. method list below
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html
    result_clustering = fclusterdata(allvectors, t=clustersize, criterion='maxclust',
                                     metric=euclidean_and_most_used_sensor_dist,
                                     method=linkage_method_for_clustering)
    """
        The linkage method to use (single, complete, average, weighted, median, centroid, ward).
        method: 
            single:     bad
            complete:   ok
            average:    bad
            weighted:   ok
            centroid:   bad
            median:     bad
            ward:       ok
    """

    # decreasing the indices by 1, from 1-15 to 0-14 to comply with other algorithms
    result_clustering = result_clustering - 1

    logger.info("end modified clustering method")

    return result_clustering


def create_sensor_relevance_matrix(allvectors, allvectors_short):
    """
    Creates a dataframe the has a sensor-list for every vector. The sensors a ordered by amount of usage.
    The lists only contain used sensors, the rest is just NaN

    @param allvectors:          A Dataframe of all vectors that should be clustered.
    @param allvectors_short:    A Dataframe of all vectors that should be clustered without the "Zero-Sensor"

    @return: Dataframe with most uses sensors
    """

    vector_length = len(allvectors_short.columns)
    # create an empty Dataframe for the results
    most_used_sensors = pd.DataFrame(index=range(1, len(allvectors) + 1), columns=range(1, vector_length))

    # iterate thought every vector
    for i in range(1, len(allvectors) + 1):
        # get the vector
        vector = allvectors_short.loc[i]
        # reset the index, so that it is just the sensor-number as integer
        vector.reset_index(drop=True, inplace=True)
        # increase by 1 because there is no sensor 0
        vector.index += 1
        # sort sensors and delete 0s
        vector = vector.sort_values(0, ascending=False)
        vector = vector[vector != 0]
        # override the value of how much the sensor is used with the actual sensor number, to get the ordered sensor list
        vector = pd.Series(data=vector.index)
        # increase by 1 because there is no sensor 0
        vector.index += 1
        # write it in the Dataframe
        most_used_sensors.loc[i, :] = vector

    return most_used_sensors
