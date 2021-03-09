##Interface section
from tkinter import filedialog
## Pm4py Section
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator


def apply_inductive_miner(log,
                          path_data_sources,
                          dir_runtime_files,
                          dir_dfg_cluster_files,
                          filename_dfg_cluster,
                          rel_proportion_dfg_threshold,
                          logging_level):
    print("Prepare inductive Miner.")

    import_file_path = filedialog.askopenfilename()
    if import_file_path:
        log = xes_importer.apply(import_file_path)
        net, initial_marking, final_marking = inductive_miner.apply(log)
        file = open(filedialog.askopenfilename(), "a")


        fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking,
                                                 variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        prec = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        gen = generalization_evaluator.apply(log, net, initial_marking, final_marking)
        simp = simplicity_evaluator.apply(net)

        file.write("Fitness: " + str(fitness) + "\n")
        file.write("Precision: " + str(prec) + "\n")
        file.write("Generalization: " + str(gen) + "\n")
        file.write("Simplicity: " + str(simp) + "\n\n")
        file.close()

        parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
        gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters, variant=pn_visualizer.Variants.FREQUENCY, log=log)
        pn_visualizer.view(gviz)
        print("Induvtive Miner - Petri Net is done.")
    else:
        print("Aborted.")


### Process tree
# from pm4py.visualization.process_tree import visualizer as pt_visualizer
#def applyInductiveMinerPt():
#    print("Prepare inductive Miner.")

#    import_file_path = filedialog.askopenfilename()
#    if import_file_path:
#        log = xes_importer.apply(import_file_path)
#        net, initial_marking, final_marking = inductive_miner.apply(log)

#        tree = inductive_miner.apply_tree(log)

#        gviz = pt_visualizer.apply(tree)
#        pt_visualizer.view(gviz)
#        print("Induvtive Miner is done.")
#    else:
#        print("Aborted.")