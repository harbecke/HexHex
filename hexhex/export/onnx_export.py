import torch

from hexhex.utils.utils import load_model


def main(model_name=None):
    if model_name is None:
        model_name = '11_2w4_1262'
    model = load_model(f"models/{model_name}.pt", export_mode=True)
    dummy_input = torch.zeros(1, 2, model.board_size, model.board_size)
    torch.onnx.export(model, dummy_input, f'{model_name}.onnx', verbose=True)


if __name__ == '__main__':
    main()
