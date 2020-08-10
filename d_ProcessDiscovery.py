import inspect
import logging
import os
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.start_activities.log import get as sa_get
from pm4py.statistics.end_activities.log import get as ea_get
from pm4py.visualization.dfg import visualizer as dfg_visualization


def create_process_models(output_case_traces_cluster, path_data_sources, dir_runtime_files, dir_dfg_cluster_files,
                          filename_dfg_cluster, min_number_of_occurrences, logging_level):
    """
    Creates directly follows graphs out of a event log.
    :param output_case_traces_cluster: traces that are visualised
    :param path_data_sources: path of sources and outputs
    :param dir_runtime_files: folder containing files read and written during runtime
    :param dir_dfg_cluster_files: folder containing dfg png files
    :param filename_dfg_cluster: filename of dfg file (per cluster)
    :param min_number_of_occurrences: threshold for filtering out sensors in dfg
    :param logging_level: level of logging
    :return:
    """

    # keep only needed columns
    output_case_traces_cluster = output_case_traces_cluster.reindex(
        columns={'Case', 'LC_Activity', 'Timestamp', 'Cluster'})
    output_case_traces_cluster = output_case_traces_cluster.rename(
        columns={'Case': 'case:concept:name', 'LC_Activity': 'concept:name',
                 'Timestamp': 'time:timestamp'})

    # create directory for dfg pngs
    os.mkdir(path_data_sources + dir_runtime_files + dir_dfg_cluster_files)
    # create dfg for each cluster
    clusters = output_case_traces_cluster.Cluster.unique()
    for cluster in clusters:
        log = output_case_traces_cluster.loc[output_case_traces_cluster.Cluster == cluster]
        log = log.astype(str)

        # convert pandas data frame to pm4py event log for further processing
        log = log_converter.apply(log)

        # keep only activities with more than certain number of occurrences
        activities = attributes_get.get_attribute_values(log, 'concept:name')
        activities = {x: y for x, y in activities.items() if y >= min_number_of_occurrences}
        log = attributes_filter.apply(log, activities)

        # create dfg out of event log
        dfg = dfg_discovery.apply(log)

        # define start and
        start_activities = sa_get.get_start_activities(log)
        end_activities = ea_get.get_end_activities(log)

        # create png of dfg (if the graph does not show a graph, it is possible that the sensors did not trigger often)
        gviz = dfg_visualization.apply(dfg=dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY,
                                       parameters={'start_activities': start_activities,
                                                   'end_activities': end_activities})
        dfg_visualization.save(gviz, path_data_sources + dir_runtime_files + dir_dfg_cluster_files + (
            filename_dfg_cluster.format(cluster=str(cluster))))

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Saved directly follows graphs into '../%s'.",
                path_data_sources + dir_runtime_files + dir_dfg_cluster_files)
