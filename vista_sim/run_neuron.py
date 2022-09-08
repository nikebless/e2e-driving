import subprocess
import sys
import shlex
import os
import gpustat


MAX_RUNS_PER_GPU = 3

def get_free_gpu_indices(max_runs_per_gpu=3):
    stats = gpustat.GPUStatCollection.new_query()
    gpu_process_counts = {}
    for gpu in stats.gpus:
        gpu_process_counts[gpu['index']] = len(gpu.processes)

    free_gpus = []
    for gpu_index, process_count in gpu_process_counts.items():
        if process_count < max_runs_per_gpu:
            free_gpus.append(gpu_index)

    return free_gpus

def get_gpu_selection_cmd(gpu_index):
    return f'CUDA_VISIBLE_DEVICES={gpu_index} CUDA_DEVICE_ORDER=PCI_BUS_ID EGL_DEVICE_ID={gpu_index}'


if __name__ == '__main__':
    '''Run vista simulation on a free GPU.

    Usage in background:
        nohup python run_neuron.py --model /path/to/model.onnx > runs/$(date +%s)_neuron.txt 2>$1 &
    
    Raw usage:
        python run_neuron.py --model /path/to/model.onnx
    '''
    free_gpus = get_free_gpu_indices()
    selected_gpu = free_gpus[0]
    gpu_selection_str = get_gpu_selection_cmd(selected_gpu)
    print(f'Using GPU {selected_gpu}.', flush=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_file = os.path.join(current_dir, 'vista_evaluate.py')
    args = " ".join(map(shlex.quote, sys.argv[1:]))
    cmd = f"{gpu_selection_str} python -u {path_to_file} {args}"

    subprocess.call(cmd, shell=True)
