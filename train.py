import os
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from configs import Config
from model import get_model
from dataset import get_train_data, get_eval_data
from optim import get_opt_schedule
from Loss import get_loss
from utils import set_log, set_seed

from eval import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='ssvh')
    parser.add_argument('--config', default='configs/conmh_act.py', type = str,
        help='config file path'
    )
    parser.add_argument('--gpu', default = 'cuda:0', type = str,
        help = 'specify gpu device'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if not os.path.exists(cfg.file_path):
        os.makedirs(cfg.file_path)

    # set log
    logger = set_log(cfg, 'log.txt')
    logger.info('Self Supervised Video Hashing Training: {}'.format(cfg.model_name))

    # set seed
    set_seed(cfg)
    logger.info('set seed: {}'.format(cfg.seed))

    logger.info('PARAMETER ......')
    logger.info(cfg)

    logger.info('loading model ......') 

    # set GPU
    args.gpu = int(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # load train data
    logger.info('loading train data ......')    
    train_loader,train_dataset = get_train_data(cfg)
    total_len = len(train_loader)

    # load eval data
    logger.info('loading eval data ......')     
    eval_loader = get_eval_data(cfg)
    epoch = 0

    # optimizer and schedule
    opt_schedule = get_opt_schedule(cfg, model)

    if cfg.use_checkpoint is not None:
        checkpoint = torch.load(cfg.use_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_schedule._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        opt_schedule._schedule.last_epoch = checkpoint['epoch']

    # mAP5
    mAP5_max=-99
    mAP5_now=-999

    while True:
        # Regular evaluation
        if cfg.dataset == 'fcv':
            if epoch % 50 == 0 and  epoch != 0:
                mAP5_now=evaluate(cfg, model, cfg.test_num_sample, logger, args, eval_loader,training=True)
        elif cfg.dataset == 'activitynet' or cfg.dataset=="ucf" or cfg.dataset=="hmdb":
            if epoch % 20 == 0 and  epoch != 0:
                mAP5_now=evaluate(cfg, model, cfg.test_num_sample, logger, args, eval_loader,training=True)

        #  Save best checkpoint -- mAP[5]
        if mAP5_now > mAP5_max:
            save_file = cfg.file_path + '/{}_{}.pth'.format(cfg.dataset, cfg.nbits,eval_loader)
            torch.save({
                'model_state_dict': model.state_dict()
            }, save_file)
            mAP5_max = mAP5_now

        logger.info('begin training stage: [{}/{}]'.format(epoch+1, cfg.num_epochs))  


        model.train()
        for i, data in enumerate(train_loader, start=1):
            opt_schedule.zero_grad()
            loss = get_loss(cfg, data, model, epoch, i, total_len, logger,args,criterion,train_dataset)
            loss.backward()
            opt_schedule._optimizer_step()
        opt_schedule._schedule_step()
        logger.info('now the learning rate is: {}'.format(opt_schedule.lr()))
        save_file = cfg.file_path + '/model.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt_schedule._optimizer.state_dict()
        }, save_file)

        epoch += 1
        if epoch >= cfg.num_epochs :
            break

if __name__ == '__main__':
    import warnings

    # Ignore specific types of warnings
    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")
    main()
