import onnx
import argparse

def convert_static_to_dynamic_bs(onnx_path_in, onnx_path_out):
    model = onnx.load(onnx_path_in)
    for inputs in model.graph.input:
        print('\nInput: {}'.format(inputs.name))
        print(inputs)
        dim1 = inputs.type.tensor_type.shape.dim[0]
        dim1.dim_param = 'batch'
    onnx.save(model, onnx_path_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert static batch size model to dynamic batch size')
    parser.add_argument('--file', type=str, help='Path to input ONNX model')
    parser.add_argument('--output', type=str, help='Path to output ONNX model')
    args = parser.parse_args()
    convert_static_to_dynamic_bs(args.file, args.output)

