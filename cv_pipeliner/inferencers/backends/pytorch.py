def get_torch_device(device=None):
    if device is not None:
        return device
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
