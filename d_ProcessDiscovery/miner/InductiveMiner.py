##Interface section
## Pm4py Section
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.objects.conversion.log import converter as log_converter


def apply_inductive_miner(log,
                          path_data_sources,
                          dir_runtime_files,
                          dir_dfg_cluster_files,
                          filename_dfg_cluster,
                          rel_proportion_dfg_threshold,
                          logging_level):
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
    log_converted = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    net, initial_marking, final_marking = inductive_miner.apply(log_converted)

    fitness = replay_fitness_evaluator.apply(log_converted, net, initial_marking, final_marking,
                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    # precision = precision_evaluator.apply(log, net, initial_marking, final_marking,
    #                                       variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    # generalization = generalization_evaluator.apply(log, net, initial_marking, final_marking)
    # simplicity = simplicity_evaluator.apply(net)

    # ToDo uncomment metrics & metrics calculation above
    metrics = {'fitness': fitness,
               #           'precision': precision,
               #           'generalization': generalization,
               #           'simplicity': simplicity
               }

    parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters,
                               variant=pn_visualizer.Variants.FREQUENCY, log=log)

    pn_visualizer.save(gviz,
                       path_data_sources + dir_runtime_files + dir_dfg_cluster_files + (str('ProcessModelIM.png')))

    pn_visualizer.view(gviz)

    return metrics
