import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
from utils.tools import l2_norm
from random import sample

import pdb

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1):
    """_summary_
    Contrastive Learning Loss
    """
    out_1 = F.normalize(out_1, dim=1)
    out_2 = F.normalize(out_2, dim=1)

    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(out_1.device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    if True:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss

def S5VH_criterion(cfg, data, model, epoch, i, total_len, logger,args,criterion=None,train_dataset=None):

    device = args.gpu
    data = {key: value.to(device) for key, value in data.items()}
    batchsize = data["full_view"].size(0)
    index = data["idx"].squeeze()

    bool_masked_pos_1 = data["mask"][:,0,:].to(device, non_blocking=True).flatten(1).to(torch.bool)
    bool_masked_pos_2 = data["mask"][:,1,:].to(device, non_blocking=True).flatten(1).to(torch.bool)

    frame_1,hash_code_1,cluster_preds_1= model(data["full_view"],bool_masked_pos_1)
    frame_2,hash_code_2,cluster_preds_2= model(data["full_view"],bool_masked_pos_2)

    # (b,nbit)
    hash_code_1 = torch.mean(hash_code_1, 1)
    hash_code_2 = torch.mean(hash_code_2, 1)


    labels_1 = data["full_view"][bool_masked_pos_1].reshape(batchsize, -1, cfg.feature_size)
    labels_2 = data["full_view"][bool_masked_pos_2].reshape(batchsize, -1, cfg.feature_size)


    mask_lr_1 = torch.mean(F.mse_loss(frame_1, labels_1,reduction="none"),dim=-1)
    mask_lr_2 = torch.mean(F.mse_loss(frame_2, labels_2,reduction="none"),dim=-1)

    # contra_loss
    contra_loss = dcl(hash_code_1, hash_code_2, batchsize, temperature=cfg.temperature, tau_plus=cfg.tau_plus)
    # recon_loss
    recon_loss = torch.mean(mask_lr_1)+torch.mean(mask_lr_2)

    if epoch < cfg.warm_up_epoch:
        cluster_preds_1 = cluster_preds_1/(cfg.nbits*cfg.temperature_cluster)
        cluster_preds_2 = cluster_preds_2/(cfg.nbits*cfg.temperature_cluster)
        # #cfg.temperature_cluster
        contra_loss_cluster = criterion(cluster_preds_1,data["cluster_label"])* 0.5 +criterion(cluster_preds_2,data["cluster_label"])* 0.5

        loss = recon_loss + cfg.alpha * contra_loss + cfg.a_cluster * contra_loss_cluster 
        if i % 50 == 0 or batchsize < cfg.batch_size:  
            logger.info('Epoch:[%d/%d] Step:[%d/%d] reconstruction_loss: %.2f contra_loss: %.2f contra_loss_cluster: %.2f' \
                % (epoch+1, cfg.num_epochs, i, total_len,\
                recon_loss.data.cpu().numpy(), contra_loss.data.cpu().numpy(),contra_loss_cluster.data.cpu().numpy()))
    else:
        with torch.no_grad():
            # Predicted Pseudo-labels
            label_preds = model.get_label_pred(data["full_view"])
            label_preds_softmax = torch.nn.functional.softmax(label_preds, dim=-1)
            # Label Smoothing Parameter
            smoothing_alpha = cfg.smoothing_alpha 
            new_labels = (1.0 - smoothing_alpha) * data["cluster_label"] + smoothing_alpha * label_preds_softmax
            # Update the Cluster Label
            for idx, item in enumerate(data["idx"]):  
                    train_dataset.update_cluster_labels(item, new_labels[idx].cpu().detach().numpy()) 

        cluster_preds_1 = cluster_preds_1/(cfg.nbits*cfg.temperature_cluster)
        cluster_preds_2 = cluster_preds_2/(cfg.nbits*cfg.temperature_cluster)
        contra_loss_cluster = criterion(cluster_preds_1,new_labels)* 0.5 +criterion(cluster_preds_2,new_labels)* 0.5
        loss = recon_loss + cfg.alpha * contra_loss + cfg.a_cluster * contra_loss_cluster 

        if i % 10 == 0:  
            logger.info('Epoch:[%d/%d] Step:[%d/%d] reconstruction_loss: %.2f contra_loss: %.2f contra_loss_cluster: %.2f' \
                % (epoch+1, cfg.num_epochs, i, total_len,\
                recon_loss.data.cpu().numpy(), contra_loss.data.cpu().numpy(),contra_loss_cluster.data.cpu().numpy()))
 
    return loss
