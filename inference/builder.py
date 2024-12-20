from .S5VH_inference import S5VH_inference

def get_inference(cfg, data, model,args):
    if cfg.model_name in ['S5VH', 'LSTM',  'Transformer', 'RWKV', 'RetNet']:
        return S5VH_inference(cfg, data, model,args)