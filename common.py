import os
import socket
import onnxruntime as ort
import time


DEVICE_ID = int(os.environ.get('CUDA_AVAILABLE_DEVICES', '0').split(',')[0])
IS_NEURON = socket.gethostname() == 'neuron'
BOLT_DIR = '/data/Bolt' if IS_NEURON else '/gpfs/space/projects/Bolt'
BAGS_DIR = os.path.join(BOLT_DIR, 'bagfiles')


class OnnxSteeringModel:
    def __init__(self, path_to_onnx_model):
        options = ort.SessionOptions()
        if IS_NEURON:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': DEVICE_ID,
                }),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # these options are necessary only for HPC, not sure why:
            # https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-876092511
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(path_to_onnx_model, options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_frame):
        return self.session.run(None, { self.input_name: input_frame })[0]


class Timing:
    def __init__(self, stats, name):
        self.stats = stats
        self.name = name
        if name not in self.stats:
            self.stats[self.name] = dict(time=0.0, count=0)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.stats[self.name]['time'] += (time.time() - self.start_time)
        self.stats[self.name]['count'] += 1