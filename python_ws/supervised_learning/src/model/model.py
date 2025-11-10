import torch
from omegaconf import DictConfig
from .lidarnet import TinyLidarNet
from .pilotnet import PilotNet

def load_model(model_cfg: DictConfig):
    model_name = model_cfg.name
    input_dim = model_cfg.input_dim
    output_dim = model_cfg.output_dim

    if model_name == "TinyLidarNet":
        model = TinyLidarNet(input_dim=input_dim, output_dim=output_dim)
    elif model_name == "PilotNet":
        model = PilotNet(num_outputs=output_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_cfg.ckpt_path:
        model.load_state_dict(torch.load(model_cfg.ckpt_path))

    return model