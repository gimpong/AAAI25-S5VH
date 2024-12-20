from .S5VH_optim import S5VH_opt_schedule

def get_opt_schedule(cfg, model):
    if cfg.model_name in ['S5VH', 'LSTM',  'Transformer', 'RWKV', 'RetNet']:
        return S5VH_opt_schedule(cfg, model)
