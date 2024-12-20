import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
import pdb

def run_kmeans(x, args,gpu_id,device):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')

    num_cluster = args.nclusters
    # for seed, num_cluster in enumerate(args.num_cluster):
    # intialize faiss clustering parameters
    d = x.shape[1]

    k = int(num_cluster)

    clus = faiss.Clustering(d, k)

    clus.verbose = False  

    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = args.seed
    clus.max_points_per_centroid = 200
    clus.min_points_per_centroid = 2

    res = faiss.StandardGpuResources()
    cfg_cluster = faiss.GpuIndexFlatConfig()
    cfg_cluster.useFloat16 = False
    cfg_cluster.device = gpu_id  
    index = faiss.GpuIndexFlatL2(res, d, cfg_cluster) 
    x = x.cpu().numpy()

    clus.train(x, index)   

    D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]
    
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
    
    # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)]          
    for im,i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])
    
    # concentration estimation (phi)        

    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)               
    
    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).to(device)

    centroids = nn.functional.normalize(centroids, p=2, dim=1)    

    im2cluster = torch.LongTensor(im2cluster).to(device) 
    return im2cluster, centroids

def compute_features(train_loader, cfg, device):
   
    output_dim = cfg.latent_dim_pca

    features = []
    for _, batch_data in enumerate(train_loader,start=1):
        data = {key: value.to(device) for key, value in batch_data.items()}
        data = data["full_view"]
        features.append(data.cpu().numpy())  

    features = np.vstack(features) # (B,N,D)
    features = np.mean(features, axis=1)

    pca = PCA(n_components=output_dim)
    reduced_features = pca.fit_transform(features)  # Nx,output_dim
    reduced_features = torch.tensor(reduced_features, device=device)

    return reduced_features

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output
    

def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
        yescnt_all[qid] = x
        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all  


def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()