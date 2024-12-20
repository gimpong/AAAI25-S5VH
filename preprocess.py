from configs import Config
import argparse
import h5py as h5
import h5py
import sklearn.decomposition as skd
from utils.preprocess.get_anchors import k_means
from utils.preprocess.calculate_neighbors import ZZ
from utils.preprocess.optimize_hash_center import getBestHash
import numpy as np
import torch
from dataset import get_train_data

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

    # Principal Component Analysis (PCA) for Dimensionality Reduction
    config = { 
        'pca_class': 'PCA', # None, 'IncrementalPCA', 'PCA'
        'pca_args': {
            'n_components': cfg.latent_dim_pca, # PCA projection dimensional 
            'whiten': False, 
            'copy': False, 
        }
    }

    with h5py.File(cfg.train_feat_path, 'r') as h5f:
        train_feat = h5f["feats"][()].mean(1)

    # train
    pca = None
    print(f"use pca?", "yes" if config['pca_class'] else 'no')
    if config['pca_class']:
        print(f"train and apply pca using {config['pca_class']}, settings: {config['pca_args']}")
        pca = getattr(skd, config['pca_class'])(**config['pca_args'])
        latent_train_feat = pca.fit_transform(train_feat)

    with h5py.File(cfg.latent_feat_path, 'w') as h5f:
        h5f.create_dataset('feats', data=latent_train_feat)

    # K-means

    h5 = h5py.File(cfg.anchor_path, 'w')

    anchors = k_means(latent_train_feat, cfg.nclusters)
    h5.create_dataset('feats',data = anchors)
    h5.close()

    '''
    We set tag=1 for closest pairs(similar),
    tag=2 for pairs with middle distances(dissimilar),
    tag = 0 for other cases (we don't care)
    '''

    Z,_,pos1 = ZZ(latent_train_feat, anchors, 3, None)
    s = np.asarray(Z.sum(0)).ravel()
    isrl = np.diag(np.power(s, -1)) 

    Adj = np.dot(np.dot(Z,isrl),Z.T)
    SS1 = (Adj>0.00001).astype('float32')

    Z,_,pos1 = ZZ(latent_train_feat, anchors, 4, None)
    s = np.asarray(Z.sum(0)).ravel()
    isrl = np.diag(np.power(s, -1)) 
    Adj = np.dot(np.dot(Z,isrl),Z.T)
    SS2 = (Adj>0.00001).astype('float32')

    Z,_,pos1 = ZZ(latent_train_feat, anchors, 5, None)
    s = np.asarray(Z.sum(0)).ravel()
    isrl = np.diag(np.power(s, -1)) 
    Adj = np.dot(np.dot(Z,isrl),Z.T)
    SS3 = (Adj>0.00001).astype('float32')

    SS4 = SS3-SS2   
    SS5 = 2*SS4+SS1
    
    hh5 = h5py.File(cfg.sim_path, 'w')
    hh5.create_dataset('adj', data = SS5)
    hh5.close()

    '''
    For each video, search several neighbors from the anchor set and save them in a file.
    To save space, we only save the index of them.
    The nearest anchor is a pseudo label of the video.
    '''  
    
    Z1,_,pos1 = ZZ(latent_train_feat, anchors, 3, None)
    h5 = h5py.File(cfg.train_assist_path, 'w')
    h5.create_dataset('pos', data = pos1)
    h5.close()

    # Calculate the Optimal Hash Centers
    train_loader , _ = get_train_data(cfg)

    # set GPU
    args.gpu = int(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    getBestHash(cfg, args, train_loader, cfg.nbits, device, initWithCSQ = cfg.initWithCSQ, rho=cfg.rho, gamma=cfg.gamma)


if __name__ == '__main__':
    import warnings

    # Ignore specific types of warnings
    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")
    main()

