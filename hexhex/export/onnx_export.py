import torch

from hexhex.utils.paths import reference_model_path
from hexhex.utils.utils import load_cpu_model


def main(model_name=None):
    if model_name is None:
        model_name = '11_2w4_2000'
    model = load_cpu_model(reference_model_path(model_name), export_mode=True)
    dummy_input = torch.zeros(1, 2, model.board_size + 2, model.board_size + 2)
    # external_data=False keeps the weights inside the .onnx file, which the frontend fetches on its own.
    torch.onnx.export(model, dummy_input, f'{model_name}.onnx', dynamo=True, external_data=False)


if __name__ == '__main__':
    main()
