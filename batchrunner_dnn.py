import os
import multiprocessing
import time
import itertools

param_dict = {
    "epochs": [2],
    "method": ["DNN"],
    "seed": range(32),
    "batch_size": [64],
    "n_ind": [200],
    "lengthscale": [2.0],
    "amplitude": [0.3],
    "temp": [1e-6],
    "viz_every": [200],
    "output": ["output/test_reproducibility_dnn"],
    "km_coeff": [0.1],
}

cmd_template = "python main.py -diag_cov -message YOLO {}"


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    instance_list = []
    for instance in itertools.product(*vals):
        instance_args = " ".join(["-{} {}".format(arg, value) for (arg, value) in zip(keys, instance)])
        instance_list.append(instance_args)
    return instance_list


commands = [cmd_template.format(instance) for instance in product_dict(**param_dict)]
# print('\n'.join(commands))
print("# experiments = {}".format(len(commands)))

gpu_list = ["''"] * 32
# gpu_list = range(8)
gpus = multiprocessing.Manager().list(gpu_list)
proc_to_gpu_map = multiprocessing.Manager().dict()


def exp_runner(cmd):
    process_id = multiprocessing.current_process().name
    if process_id not in proc_to_gpu_map:
        proc_to_gpu_map[process_id] = gpus.pop()
        print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))

    # print(cmd + ' -gpu {}'.format(proc_to_gpu_map[process_id]))
    return os.system(cmd + " -gpu {}".format(proc_to_gpu_map[process_id]))


p = multiprocessing.Pool(processes=len(gpus))
rets = p.map(exp_runner, commands)
print(rets)
