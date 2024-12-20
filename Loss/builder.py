from .S5VH_loss import S5VH_criterion

def get_loss(cfg, data, model, epoch, i, total_len, logger,args,criterion = None,dataset = None):
    if cfg.model_name in ['S5VH', 'LSTM',  'Transformer', 'RWKV', 'RetNet']:
        return S5VH_criterion(cfg, data, model, epoch, i, total_len, logger,args,criterion,train_dataset = dataset)
