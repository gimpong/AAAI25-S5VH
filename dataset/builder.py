from .S5VH_dataset import get_S5VH_train_loader, get_S5VH_eval_loader

def get_train_data(cfg):
    if cfg.model_name in ['S5VH', 'LSTM',  'Transformer', 'RWKV', 'RetNet']:
        return get_S5VH_train_loader(cfg, shuffle=True)

def get_eval_data(cfg):
    if cfg.model_name in ['S5VH', 'LSTM',  'Transformer', 'RWKV', 'RetNet']:
        return get_S5VH_eval_loader(cfg)