import torch
from .S5VH import S5VH

def get_model(cfg):
    if cfg.model_name in ['S5VH', 'LSTM',  'Transformer', 'RWKV', 'RetNet']:
        model = S5VH(cfg)
    return model
    