import io
import os
import time
import h5py
import numpy as np
import scipy.io as sio
import logging
import argparse
import torch
from torch.autograd import Variable
from configs import Config
from model import get_model
from dataset import get_eval_data
from inference import get_inference
from utils import mAP, set_log, set_seed



def parse_args():
    parser = argparse.ArgumentParser(description='ssvh')
    parser.add_argument('--config', default='configs/conmh_fcv.py', type = str,
        help='config file path'
    )
    
    parser.add_argument('--gpu', default = '0', type = str,
        help = 'specify gpu device'
    )

    args = parser.parse_args()
    return args


class Array():
    def __init__(self):
        pass

    def setmatrcs(self, matrics):
        self.matrics = matrics

    def concate_v(self, matrics):
        self.matrics = np.vstack((self.matrics,matrics))

    def getmatrics(self):
        return self.matrics


class Monitor:
    def __init__(self, max_patience=5, delta=1e-6):
        self.counter_ = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta

    def update(self, cur_value):
        self.counter_ += 1
        is_break = False
        is_lose_patience = False
        if cur_value < self.best_value + self.delta:
            cur_value = 0
            self.patience -= 1
            logging.info("the monitor loses its patience to %d!" % self.patience)
            is_lose_patience = True
            if self.patience == 0:
                self.patience = self.max_patience
                is_break = True
        else:
            self.patience = self.max_patience
            self.best_value = cur_value
            cur_value = 0
        return is_break, is_lose_patience

    @property
    def counter(self):
        return self.counter_


def evaluate(cfg, model, num_sample, logger,args,eval_loader,training=False):

    model.eval()

    logger.info('eval data number: {}'.format(num_sample))
    hashcode = np.zeros((num_sample, cfg.nbits), dtype = np.float32)
    label_array = Array()
    rem = num_sample % cfg.test_batch_size
    eval_loader.dataset.set_mode('test')
    for i, one_label_path in enumerate(cfg.label_path):
        if i == 0:
            if cfg.dataset == 'activitynet':
                labels = sio.loadmat(one_label_path)['re_label']
            else:
                labels = sio.loadmat(one_label_path)['labels']
        else:
            labels = np.concatenate((labels, sio.loadmat(one_label_path)['labels']), axis=0)

    label_array.setmatrcs(labels)
    
    batch_num = len(eval_loader)
    time0 = time.time()
    for i, data in enumerate(eval_loader):
        BinaryCode = get_inference(cfg, data, model,args)

        if i == batch_num - 1:
            hashcode[i*cfg.test_batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
        else:
            hashcode[i*cfg.test_batch_size:(i+1)*cfg.test_batch_size,:] = BinaryCode.data.cpu().numpy()

    test_hashcode = np.matrix(hashcode)

    if cfg.dataset == 'fcv' and training:
        n_query = 4000
        time1 = time.time()
        logger.info('retrieval costs: {}'.format(time1 - time0))
        Hamming_distance = 0.5 * (-np.dot(test_hashcode, test_hashcode.transpose()) + cfg.nbits)
        time2 = time.time()
        logger.info('hamming distance computation costs: {}'.format(time2 - time1))
        HammingRank = np.argsort(Hamming_distance, axis=0)
        time3 = time.time()
        logger.info('hamming ranking costs: {}'.format(time3 - time2))

        labels = label_array.getmatrics()
        logger.info('labels shape: {}'.format(labels.shape))
        sim_labels = np.dot(labels, labels.transpose())
        time6 = time.time()
        logger.info('similarity labels generation costs: {}'.format(time6 - time3))
    elif cfg.dataset == 'fcv':
        # test_hashcode: NxB, query_hashcode: QxB
        query_hashcode = test_hashcode[:n_query]
        time1 = time.time()
        logger.info('retrieval costs: {}'.format(time1 - time0))
        
        Hamming_distance = 0.5 * (-np.dot(test_hashcode, query_hashcode.transpose()) + cfg.nbits) # NxQ
        time2 = time.time()
        logger.info('hamming distance computation costs: {}'.format(time2 - time1))
        HammingRank = np.argsort(Hamming_distance, axis=0) # NxQ
        time3 = time.time()
        logger.info('hamming ranking costs: {}'.format(time3 - time2))

        labels = label_array.getmatrics()
        query_labels = labels[:n_query]
        logger.info('labels shape: {}'.format(labels.shape))
        
        sim_labels = np.dot(labels, query_labels.transpose()) # NxQ
        time6 = time.time()
        logger.info('similarity labels generation costs: {}'.format(time6 - time3))
    elif cfg.dataset in ['activitynet', 'hmdb', 'ucf']:
        logger.info('loading query data ......') 
        query_hashcode = np.zeros((cfg.query_num_sample, cfg.nbits), dtype = np.float32)
        query_label_array = Array()
        query_rem = cfg.query_num_sample % cfg.test_batch_size
        eval_loader.dataset.set_mode('query')
        for i, one_label_path in enumerate(cfg.query_label_path):
            if i == 0:
                if cfg.dataset == 'activitynet':
                    query_labels = sio.loadmat(one_label_path)['q_label']
                else:
                    query_labels = sio.loadmat(one_label_path)['labels']
            else:
                query_labels = np.concatenate((query_labels, sio.loadmat(one_label_path)['labels']), axis=0)
        query_label_array.setmatrcs(query_labels)
        batch_num = len(eval_loader)
        for i, data in enumerate(eval_loader):
            query_BinaryCode = get_inference(cfg, data, model,args)
            if i == batch_num - 1:
                query_hashcode[i*cfg.test_batch_size:,:] = query_BinaryCode[:query_rem,:].data.cpu().numpy()
            else:
                query_hashcode[i*cfg.test_batch_size:(i+1)*cfg.test_batch_size,:] = \
                                                    query_BinaryCode.data.cpu().numpy()

        query_hashcode = np.matrix(query_hashcode)
        time1 = time.time()
        logger.info('retrieval costs: {}'.format(time1 - time0))
        Hamming_distance = 0.5 * (-np.dot(test_hashcode, query_hashcode.transpose()) + cfg.nbits)

        time2 = time.time()
        logger.info('hamming distance computation costs: {}'.format(time2 - time1))
        HammingRank = np.argsort(Hamming_distance, axis=0)
        time3 = time.time()
        logger.info('hamming ranking costs: {}'.format(time3 - time2))
        
        query_labels = query_label_array.getmatrics()
        labels = label_array.getmatrics()
        logger.info('labels shape: {} and {}'.format(query_labels.shape, labels.shape))
        sim_labels = np.dot(labels, query_labels.transpose())
        time6 = time.time()
        logger.info('similarity labels generation costs: {}'.format(time6 - time3))
    
    maps = []
    map_list = [5,20,40,60,80,100]
    for i in map_list:
        map, _, _ = mAP(sim_labels, HammingRank, i)
        maps.append(map)
        logger.info('topK: {}:, map: {}'.format(i, map))
    time7 = time.time()

    # return mAP[5]
    return maps[0]
    # return (torch.tensor(sim_labels.transpose()).cuda().cpu(),torch.tensor(Hamming_distance.transpose()).cuda().cpu())

def Get_data(cfg, model, num_sample, logger,args,eval_loader):
    model.eval()
    logger.info('eval data number: {}'.format(num_sample))
    hashcode = np.zeros((num_sample, cfg.nbits), dtype = np.float32)
    label_array = Array()
    rem = num_sample % cfg.test_batch_size
    eval_loader.dataset.set_mode('test')
    for i, one_label_path in enumerate(cfg.label_path):
        if i == 0:
            if cfg.dataset == 'activitynet':
                labels = sio.loadmat(one_label_path)['re_label']
            else:
                labels = sio.loadmat(one_label_path)['labels']
        else:
            labels = np.concatenate((labels, sio.loadmat(one_label_path)['labels']), axis=0)
    print(f"labels shape :{labels.shape}")
    label_array.setmatrcs(labels) #加载label
    
    batch_num = len(eval_loader)
    time0 = time.time()
    for i, data in enumerate(eval_loader):
        BinaryCode = get_inference(cfg, data, model,args)

        if i == batch_num - 1:
            hashcode[i*cfg.test_batch_size:,:] = BinaryCode[:rem,:].data.cpu().numpy()
        else:
            hashcode[i*cfg.test_batch_size:(i+1)*cfg.test_batch_size,:] = BinaryCode.data.cpu().numpy()

    # Obtain the hash code of the test set database
    test_hashcode = np.matrix(hashcode)
    print(f"test_hashcode shape :{test_hashcode.shape}")
    if cfg.dataset in ['yfcc', 'activitynet', 'hmdb', 'ucf']:
        logger.info('loading query data ......') 
        query_hashcode = np.zeros((cfg.query_num_sample, cfg.nbits), dtype = np.float32)
        query_label_array = Array()
        query_rem = cfg.query_num_sample % cfg.test_batch_size
        eval_loader.dataset.set_mode('query')
        for i, one_label_path in enumerate(cfg.query_label_path):
            if i == 0:
                if cfg.dataset == 'activitynet':
                    query_labels = sio.loadmat(one_label_path)['q_label']
                else:
                    query_labels = sio.loadmat(one_label_path)['labels']
            else:
                query_labels = np.concatenate((query_labels, sio.loadmat(one_label_path)['labels']), axis=0)
        print(f"query labels shape :{query_labels.shape}")
        # Obtain the hash code of the query set database
        query_label_array.setmatrcs(query_labels)
        batch_num = len(eval_loader)
        for i, data in enumerate(eval_loader):
            query_BinaryCode = get_inference(cfg, data, model,args)
            if i == batch_num - 1:
                query_hashcode[i*cfg.test_batch_size:,:] = query_BinaryCode[:query_rem,:].data.cpu().numpy()
            else:
                query_hashcode[i*cfg.test_batch_size:(i+1)*cfg.test_batch_size,:] = \
                                                    query_BinaryCode.data.cpu().numpy()

        query_hashcode = np.matrix(query_hashcode) 
        # print(f"query hash code shape :{query_hashcode.shape}")
        Hamming_distance = 0.5 * (-np.dot(test_hashcode, query_hashcode.transpose()) + cfg.nbits)
        # print(f"Hamming_distance shape : {Hamming_distance.shape}")
        HammingRank = np.argsort(Hamming_distance, axis=0)
        query_labels = query_label_array.getmatrics()
        labels = label_array.getmatrics()
        sim_labels = np.dot(labels, query_labels.transpose())
        scores = Hamming_distance.transpose()
        labels_t = sim_labels.transpose()
        # print(f"sorces shape :{scores.shape},labels shape {labels.shape}")
        return torch.tensor(scores).to("cuda:3"),torch.tensor(labels_t).to("cuda:3"),test_hashcode,labels,query_hashcode,query_labels


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set logging
    logger = set_log(cfg, 'map.txt')
    logger.info('Self Supervised Video Hashing Evaluation: {}'.format(cfg.model_name))

    # set seed
    set_seed(cfg)
    logger.info('set seed: {}'.format(cfg.seed))

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    logger.info('PARAMETER ......')
    logger.info(cfg)    

    args.gpu = (int)(args.gpu)
    logger.info('loading model ......') 
    model = get_model(cfg).to(args.gpu)
    
    checkpoint = torch.load(cfg.file_path + '/{}_{}.pth'.format(cfg.dataset, cfg.nbits))
    model.load_state_dict(checkpoint['model_state_dict'])

    num_sample = cfg.test_num_sample
    eval_loader = get_eval_data(cfg)
    evaluate(cfg, model, num_sample, logger,args,eval_loader)
    # tensors = evaluate(cfg, model, num_sample, logger,args,eval_loader)
    # torch.save(tensors,f"./PRdata/Final-{cfg.dataset}-{cfg.nbits}.pt")
    # _,_,test_hashcode,labels,query_hashcode,query_labels= Get_data(cfg, model, num_sample, logger,args,eval_loader)
        
    # np.savez("./Final-ucf-64.npz",test_hashcode=test_hashcode.A,labels = labels,query_hashcode = query_hashcode.A,query_labels = query_labels)