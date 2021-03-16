from subprocess import Popen, PIPE, STDOUT

def compute_entropia(variant, xesFile, pnmlFile):

    args = ['java', '-jar', 'jbpt-pm-entropia-1.6.jar', f'-{variant}', '-srel=3', '-sret=3', f'-rel={xesFile}', f'-ret={pnmlFile}', '--silent'] 
    p = Popen(args, stdout=PIPE, stderr=STDOUT)

    res = ""
    for line in p.stdout:
        res = res + str(line)
    
    return res


precision = compute_entropia("pmp", xesFile = "log_export.xes", pnmlFile ="petri_net.pnml")
print(f'Precision {precision}')

recall = compute_entropia("pmr", xesFile = "log_export.xes", pnmlFile ="petri_net.pnml")
print(f'Recall {recall}')