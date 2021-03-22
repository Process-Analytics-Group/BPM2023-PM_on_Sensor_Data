import inspect
import logging

from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
from b_ActivityDiscovery.self_organizing_map.sompy import SOMFactory
import numpy as np
from u_utils import u_helper as helper
import z_setting_parameters as settings


def cluster_mod_1(allvectors, dict_distance_adjacency_sensor, vectorization_type, clustersize=15,
                  linkage_method_for_clustering='ward'):
    '''
    Clusters the dataset using custom calculation.

    @param allvectors:                      A lIst of all vectors that should be clustered.
    @param dict_distance_adjacency_sensor:  A Dictionary with a "distance_matrix" inside
    @param vectorization_type:              Specifies if the vector is only time or sensor quantity or if its both
    @param clustersize:                     Specifies the number of cluster. Default is 15.
    @param linkage_method_for_clustering:   Specifies the linkage method. Default is 'ward'. The linkage method to use
                                            (single, complete, average, weighted, median centroid, ward)

    @return:                                result list returns cluster for each vector
    '''

    # distance calculation
    def mydist(v1, v2):
        '''
        distance calculation between two vectors that is used for clustering.

        1. The Euclidean distance between the Vectors is calculated.
        2. The most used Sensor of each vector are identified. Than the needed steps over
           neighbouring sensors, between this two sensors, are counted.
        3. Euclidean distance * needed steps = calculated distance between two vectors

        @param v1: first vector
        @param v2: second vector
        @return: calculated distance between two vectors
        '''

        # prints progress as % (every 500.000 calculations)
        # total number of calculations: 88571395

        """
        global count
        if count % 500000 == 0:
            print(str((count / 88571395) * 100) + " %")
        count += 1
        """

        # finding the most used sensor in vector 1 & 2

        # Transforming the np array into a python list. Working on a standard python list has proven to be faster in this case.
        l1 = list(v1)
        l2 = list(v2)

        vector_length = len(l1)

        # special case for 'quantity_time'. Most used sensor is determent by time. (Second half of array)
        # first entry of the vector is not used, because it is the "no sensor" time
        if vectorization_type == 'quantity_time':
            l1_short = l1[((vector_length // 2) + 1):]
            l2_short = l2[((vector_length // 2) + 1):]
        else:
            l1_short = l1[1:]
            l2_short = l2[1:]

        # getting the index of the most used sensor
        main_sensor_v1 = l1_short.index(max(l1_short)) + 1
        main_sensor_v2 = l2_short.index(max(l2_short)) + 1

        # euclidean distance * Steps between most used sensors
        return np.linalg.norm(v1 - v2) * dict_distance_adjacency_sensor["distance_matrix"][main_sensor_v1][main_sensor_v2]

    # clustering method that can use a custom distance calculation. 'maxclust' and t=15 -> 15 cluster. method list below
    result_clustering = fclusterdata(allvectors, t=clustersize, criterion='maxclust', metric=mydist,
                                     method=linkage_method_for_clustering)
    """
        The linkage method to use (single, complete, average, weighted, median centroid, ward).
        method: 
            singel:     bad
            complete:   ok
            average:    bad
            weighted:   ok
            centroid:   bad
            median:     bad
            ward:       ok
    """

    # decreasing the indices by 1, from 1-15 to 0-14 to comply with other algorithms
    for i in range(0, len(result_clustering)):
        result_clustering[i] = result_clustering[i] - 1

    return result_clustering
