import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

# import csv
# ToDo: handover from other module directly
log_csv = pd.read_csv('/Users/dominik-cau/Documents/Lernen/Uni/Promotion/Python/'
                      '2020-02-18_ProcessModelDiscovery/Data-Sources/runtime-files/'
                      '2020-06-15_22-08-51/Cases_Cluster.csv',
                      sep=';')
# log_csv = log_csv.drop(columns=['Activity', 'Sensor_Added', 'Duration', 'LC', 'Person', 'BMU', 'kMeansVanilla'])
log_csv = log_csv.loc[(log_csv.Cluster == 3)]
log_csv = log_csv.reindex(columns=['Case', 'LC_Activity', 'Timestamp'])
log_csv = log_csv.astype(str)
log_csv = log_csv.rename(columns={'Case': "case:concept:name", 'LC_Activity': 'concept:name', 'Timestamp': 'timestamp'})

log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('timestamp')
event_log = log_converter.apply(log_csv)

from pm4py.algo.filtering.log.attributes import attributes_filter
event_log = attributes_filter.apply_numeric_events(event_log, 0, 800,
                                                   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "amount"})

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
dfg = dfg_discovery.apply(event_log)

from pm4py.visualization.dfg import visualizer as dfg_visualization
gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)

print("")
