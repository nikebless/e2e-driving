import os
import socket
import onnxruntime as ort

IS_NEURON = socket.gethostname() == 'neuron'
BOLT_DIR = '/data/Bolt' if IS_NEURON else '/gpfs/space/projects/Bolt'
BAGS_DIR = os.path.join(BOLT_DIR, 'bagfiles')

LEXUS_LENGTH = 4.89
LEXUS_WIDTH = 1.895
LEXUS_WHEEL_BASE = 2.79
LEXUS_STEERING_RATIO = 14.7

FULL_IMAGE_WIDTH = 1928
FULL_IMAGE_HEIGHT = 1208

IMAGE_CROP_XMIN = 300
IMAGE_CROP_XMAX = 1620
IMAGE_CROP_YMIN = 520
IMAGE_CROP_YMAX = 864

class OnnxModel:
    def __init__(self, path_to_onnx_model):
        options = ort.SessionOptions()
        if not IS_NEURON:
            # these options are necessary only for HPC, not sure why:
            # https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-876092511
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path_to_onnx_model, options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_frame):
        return self.session.run(None, { self.input_name: input_frame })[0]