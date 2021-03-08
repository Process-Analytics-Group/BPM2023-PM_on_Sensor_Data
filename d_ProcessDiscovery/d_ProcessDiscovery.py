import inspect
import logging
import os
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.start_activities.log import get as sa_get
from pm4py.statistics.end_activities.log import get as ea_get
from pm4py.visualization.dfg import visualizer as dfg_visualization
from d_ProcessDiscovery.HeuristicMiner import applyHeuristicMiner


def create_activtiy_models(output_case_traces_cluster, path_data_sources, dir_runtime_files, dir_dfg_cluster_files,
                           filename_dfg_cluster, rel_proportion_dfg_threshold, logging_level):
    """
    Creates directly follows graphs out of an event log.
    :param output_case_traces_cluster: traces that are visualised
    :param path_data_sources: path of sources and outputs
    :param dir_runtime_files: folder containing files read and written during runtime
    :param dir_dfg_cluster_files: folder containing dfg png files
    :param filename_dfg_cluster: filename of dfg file (per cluster)
    :param rel_proportion_dfg_threshold: threshold for filtering out sensors in dfg relative to max occurrences of a sensor
    :param logging_level: level of logging
    :return:
    """

    # keep only needed columns
    output_case_traces_cluster = output_case_traces_cluster.reindex(
        columns={'Case', 'LC_Activity', 'Timestamp', 'Cluster'})
    output_case_traces_cluster = output_case_traces_cluster.rename(
        columns={'Case': 'case:concept:name',
                 'LC_Activity': 'concept:name',
                 'Timestamp': 'time:timestamp'})

    # create directory for dfg pngs
    os.mkdir(path_data_sources + dir_runtime_files + dir_dfg_cluster_files)
    # create dfg for each cluster
    clusters = output_case_traces_cluster.Cluster.unique()
    for cluster in clusters:
        log = output_case_traces_cluster.loc[output_case_traces_cluster.Cluster == cluster]
        log = log.astype(str)
        log['time:timestamp'] = pd.to_datetime(output_case_traces_cluster['time:timestamp'])
        # convert pandas data frame to pm4py event log for further processing
        log = log_converter.apply(log)

        # keep only activities with more than certain number of occurrences
        activities = attributes_get.get_attribute_values(log, 'concept:name')
        # determine that number relative to the max number of occurrences of a sensor in a cluster. (the result is
        # the threshold at which an activity/activity strand is kept)
        min_number_of_occurrences = round((max(activities.values()) * rel_proportion_dfg_threshold), 0)
        activities = {x: y for x, y in activities.items() if y >= min_number_of_occurrences}
        log = attributes_filter.apply(log, activities)

        # create dfg out of event log
        dfg = dfg_discovery.apply(log)

        # define start and
        start_activities = sa_get.get_start_activities(log)
        end_activities = ea_get.get_end_activities(log)


        # create png of dfg (if the graph does not show a graph, it is possible that the sensors did not trigger often)
        gviz = dfg_visualization.apply(dfg0=dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY,
                                       parameters={'start_activities': start_activities,
                                                   'end_activities': end_activities})
        dfg_visualization.save(gviz, path_data_sources + dir_runtime_files + dir_dfg_cluster_files + (
            filename_dfg_cluster.format(cluster=str(cluster))))
##########################################################
        # from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
        # net, im, fm = heuristics_miner.apply(log, parameters={
        #     heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})
        #
        # from pm4py.visualization.petrinet import visualizer as pn_visualizer
        # gviz_pn = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz_pn)
        #
        # from pm4py.evaluation.generalization import evaluator as generalization_evaluator
        # gen = generalization_evaluator.apply(log, net, im, fm)
#################################################################################
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Saved directly follows graphs into '../%s'.",
                path_data_sources + dir_runtime_files + dir_dfg_cluster_files)


def create_process_model(output_case_traces_cluster, path_data_sources, dir_runtime_files, dir_dfg_cluster_files,
                         filename_dfg_cluster, rel_proportion_dfg_threshold, logging_level):

    metrics = applyHeuristicMiner(log=output_case_traces_cluster,
                                  path_data_sources=path_data_sources,
                                  dir_runtime_files=dir_runtime_files,
                                  dir_dfg_cluster_files=dir_dfg_cluster_files,
                                  filename_dfg_cluster=filename_dfg_cluster,
                                  rel_proportion_dfg_threshold=rel_proportion_dfg_threshold,
                                  logging_level=logging_level)

    return metrics
