import multiprocessing as mp
import pdb
import pickle
import random
from multiprocessing import Process, Queue

import h5py
import numpy as np
import torch
import torch.utils.data as data



# Random Mask
class RandomMaskingGenerator:
    """_summary_
    Random masking process on the dataset.
    """
    def __init__(self, max_frames, mask_ratio):
        self.num_patches = max_frames
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        idx = [i for i in range(self.num_patches)]
        random.shuffle(idx)
        idx1 = idx[:(self.num_patches - self.num_mask)]
        idx2 = idx[-(self.num_patches - self.num_mask):]
        mask1 = np.ones(self.num_patches)
        mask2 = np.ones(self.num_patches)
        mask1[idx1] = 0.
        mask2[idx2] = 0.
        return [mask1, mask2]

class TrainDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        with h5py.File(cfg.train_assist_path,'r') as h5_file:
            self.neighbor = h5_file['pos'][:]     
        
        with h5py.File(cfg.anchor_path,'r') as h5_file:
            self.achors = h5_file['feats'][:]  
        
        with h5py.File(cfg.train_feat_path, 'r') as h5_file:
            self.video_feats = h5_file['feats'][:]
        
        self.maskgenerator = RandomMaskingGenerator(cfg.max_frames, cfg.mask_ratio)

        # Initialize pseudo labels as one-hot vectors
        self.cluster_labels = np.eye(cfg.nclusters)[self.neighbor[:, 0]]

    def update_cluster_labels(self, item, new_labels):
        self.cluster_labels[item] = new_labels


    def __getitem__(self, item):
        t1 = self.video_feats[item]    
        visual_word = t1
        
        mask = self.maskgenerator()
        cluster_label = self.cluster_labels[item]  
        # cluster_label = 0
        output = {"full_view": visual_word,"idx":item,"cluster_label":cluster_label,"mask": mask}
        return {key: torch.as_tensor(value) for key, value in output.items()}

    def __len__(self):
        return len(self.video_feats)


class TestDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = 'test'

        for i, one_feature_h5_path in enumerate(cfg.test_feat_path):
            with h5py.File(one_feature_h5_path, 'r') as h5_file:
                if i == 0:
                    self.video_feats = h5_file['feats'][:]
                else:
                    self.video_feats = np.concatenate((self.video_feats, h5_file['feats'][:]), \
                                                        axis=0)

        if cfg.dataset in ['activitynet', 'hmdb', 'ucf']:
            for i, one_feature_h5_path in enumerate(cfg.query_feat_path):
                with h5py.File(one_feature_h5_path, 'r') as h5_file:
                    if i == 0:
                        self.query_feats = h5_file['feats'][:]
                    else:
                        self.query_feats = np.concatenate((self.query_feats, h5_file['feats'][:]), \
                                                        axis=0) 


    def __getitem__(self, item):
        if self.mode == 'test':
            visual_word = self.video_feats[item]
        elif self.mode == 'query':
            visual_word = self.query_feats[item]

        output = {"full_view": visual_word}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __len__(self):
        if self.mode == 'test':
            return len(self.video_feats)
        elif self.mode == 'query':
            return len(self.query_feats)

    def set_mode(self, mode):
        self.mode = mode
        assert self.mode in ['test', 'query'], 'unknown eval mode'


def get_S5VH_train_loader(cfg, shuffle=True, num_workers=1, pin_memory=True):
    batch_size = cfg.batch_size
    num_workers = cfg.workers

    v = TrainDataset(cfg)
    data_loader = torch.utils.data.DataLoader(dataset=v,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
    return data_loader,v


def get_S5VH_eval_loader(cfg, shuffle=False, num_workers=1, pin_memory=False):
    batch_size = cfg.test_batch_size
    num_workers = cfg.workers
    
    vd = TestDataset(cfg)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
    return data_loader

