import sys
import torch


def num_trainable_parameters(model):
    parameters = [param for param in model.parameters() if param.requires_grad]
    n_params = sum([param.numel() for param in parameters])
    return n_params


if __name__ == "__main__":
    model_name = sys.argv[1]
    model = torch.load(f"saved_models/{model_name}")
    print(model)
    print(num_trainable_parameters(model))
