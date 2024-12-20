import torch
import pdb

def S5VH_inference(cfg, data, model,args):
    data = {key: value.to(args.gpu) for key, value in data.items()}

    my_H = model.inference(data["full_view"], cfg.model_name)
    my_H = torch.mean(my_H, 1)
    
    BinaryCode = torch.sign(my_H)
    return BinaryCode