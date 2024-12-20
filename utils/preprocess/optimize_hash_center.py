import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import random
from scipy.linalg import hadamard, eig
import copy
import gc
import os
import time
from tqdm import tqdm
import json
import scipy.sparse.linalg as linalg
from scipy.sparse import csc_matrix
import copy
import time
import torch
from utils import run_kmeans,compute_features
import pdb
import h5py
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
# Hamming distance to inner product <> = bit-2d
# inner product to Hamming distance d = 1/2(bit-<>)

def get_margin(bit, n_class):

    L = bit

    right = (2 ** L) / n_class

    d_min = 0
    d_max = 0

    for j in range(2 * L + 4):
        dim = j
        sum_1 = 0
        sum_2 = 0
        for i in range((dim - 1) // 2 + 1):
            sum_1 += comb(L, i)
        for i in range((dim) // 2 + 1):
            sum_2 += comb(L, i)
        if sum_1 <= right and sum_2 > right:
            d_min = dim
    for i in range(2 * L + 4):
        dim = i
        sum_1 = 0
        sum_2 = 0
        for j in range(dim):
            sum_1 += comb(L, j)
        for j in range(dim - 1):
            sum_2 += comb(L, j)
        if sum_1 >= right and sum_2 < right:
            d_max = dim
            break

    alpha_neg = L - 2 * d_max

    alpha_pos = L

    return d_max, d_min


def CSQ_init(n_class, bit):
    """
    Hadamard Matrix for Hash Center Initialization
    """
    h_k = hadamard(bit)
    h_2k = np.concatenate((h_k, -h_k), 0)
    hash_center = h_2k[:n_class]

    if h_2k.shape[0] < n_class:
        hash_center = np.resize(hash_center, (n_class, bit))
        for k in range(10):
            for index in range(h_2k.shape[0], n_class):
                ones = np.ones(bit)
                ones[random.sample(list(range(bit)), bit // 2)] = -1
                hash_center[index] = ones
            c = []
            for i in range(n_class):
                for j in range(i, n_class):
                    c.append(sum(hash_center[i] != hash_center[j]))
            c = np.array(c)
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                break
    return hash_center


def init_hash(n_class, bit):
    """
    "Random initialization of hash centers"
    """
    hash_centers = -1 + 2 * np.random.random((n_class, bit))
    hash_centers = np.sign(hash_centers)
    return hash_centers



def cal_Cx(x, H):
    return np.dot(H, x)



def cal_M(H):
    return np.dot(H.T, H) / H.shape[0]



def cal_b(H):
    return np.dot(np.ones(H.shape[0], dtype=np.float64), H) / H.shape[0]



def cal_one_hamm(b, H):
    temp = 0.5 * (b.shape[0] - np.dot(H, b))
    return temp.mean() + temp.min(), temp.min()



def cal_hamm(H):
    """
    This function is used to calculate statistical information about the Hamming distances between hash codes.
    """

    dist = []
    for i in range(H.shape[0]):
        for j in range(i + 1, H.shape[0]):
            TF = np.sum(H[i] != H[j])
            dist.append(TF)
    dist = np.array(dist)
    st = dist.sum()

    return st, dist.mean(), dist.min(), dist.var(), dist.max()

def cos_simi(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def eval_metrics(H,W):
    hamming_distance, _, _, _, _ = cal_hamm(H)
    
    K = H.shape[0]
    Semantic_Consistency = 0
    for i in range(H.shape[0]):
        for j in range(i + 1, H.shape[0]):

            Semantic_Consistency += cos_simi(H[i],H[j]) - cos_simi(W[i],W[j])

    return hamming_distance/(K*K) - Semantic_Consistency


def in_range(z1, z2, z3, bit):
    flag = True
    for item in z1:
        if item < -1 and item > 1:
            flag = False
            return flag
    for item in z3:
        if item < 0:
            flag = False
            return flag
    res = 0
    for item in z2:
        res += item ** 2
    if abs(res - bit) > 0.001:
        flag = False
        return flag
    return flag



def get_min(b, H):
    temp = []
    for i in range(H.shape[0]):
        TF = np.sum(b != H[i])
        temp.append(TF)
    temp = np.array(temp)
    return temp.min()

def Lp_box_one(b, H, d_max, n_class, bit, rho, gamma, W_ex, Wi):

    b = b.astype(np.float64)
    H = H.astype(np.float64)

    d = bit - 2 * d_max

    Wei_ = np.dot(W_ex, Wi)
    Wei_mean = np.mean(Wei_)
    Wei_ -= Wei_mean
    Wei = -bit + (Wei_ - min(Wei_)) / (max(Wei_) - min(Wei_)) * (bit + bit)
    Wei = Wei.astype(np.float64)

    M = cal_M(H)  # n x n
    C = cal_b(H)  # n x 1
    out_iter = 10000
    in_iter = 10
    upper_rho = 1e9
    learning_fact = 1.07
    count = 0
    best_eval, best_min = cal_one_hamm(np.sign(b), H)
    best_B = b

    z1 = b.copy()
    z2 = b.copy()
    z3 = d - cal_Cx(np.sign(b), H)
    y1 = np.random.rand(bit)
    y2 = np.random.rand(bit)
    y3 = np.random.rand(n_class - 1)

    z1 = z1.astype(np.float64)
    z2 = z2.astype(np.float64)
    z3 = z3.astype(np.float64)
    y1 = y1.astype(np.float64)
    y2 = y2.astype(np.float64)
    y3 = y3.astype(np.float64)
    alpha = 1.0

    for e in range(out_iter):
        for ei in range(in_iter):

            left = ((rho + rho) * np.eye(bit, dtype=np.float64) + (rho + 2 * alpha) * np.dot(H.T, H))
            left = left.astype(np.float64)
            right = (rho * z1 + rho * z2 + rho * np.dot(H.T, (d - z3)) - y1 - y2 - np.dot(H.T,
                                                                                          y3) - C + 2 * alpha * np.dot(
                H.T, Wei))
            right = right.astype(np.float64)
            b = np.dot(np.linalg.inv(left), right)

            z1 = b + 1 / rho * y1

            z2 = b + 1 / rho * y2

            z3 = d - np.dot(H, b) - 1 / rho * y3

            if in_range(z1, z2, z3, bit):
                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)
                break
            else:
                z1[z1 > 1] = 1
                z1[z1 < -1] = -1

                norm_x = np.linalg.norm(z2)
                z2 = np.sqrt(bit) * z2 / norm_x

                z3[z3 < 0] = 0

                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)

        rho = min(learning_fact * rho, upper_rho)
        if rho == upper_rho:
            count += 1
            eval, mini = cal_one_hamm(np.sign(b), H)
            if eval > best_eval:
                best_eval = eval
                best_min = mini
                best_B = np.sign(b)
        if count == 100:
            # best_B = np.sign(b)
            break

    # best_B = np.sign(b)
    return best_B, H



def Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, best_st, W):
    """
    Optimize the hash center matrix B to improve the quality of hash codes and iteratively find the optimal hash centers.
    The function updates the hash center matrix B through multiple iterations and stops optimization when certain conditions are met, returning the best hash center matrix best_B.
    """
    count = 0
    for oo in range(20):
        for i in range(n_class):
            # H is the hash matrix excluding the i-th hash center
            H = np.vstack((B[:i], B[i + 1:]))  # m-1 x n
            # W_ex is the weight matrix excluding the weights of the i-th category
            W_ex = np.vstack((W[:i], W[i + 1:]))
            # Wi is the weight vector of the i-th category
            Wi = W[i]
            # Update the hash center of the current category based on the hash centers of other categories and their associated weights
            B[i], _ = Lp_box_one(B[i], H, d_max, n_class, bit, rho, gamma, W_ex, Wi)
        
        eval_st = eval_metrics(B,W)

        if eval_st > best_st:
            best_st = eval_st
            best_B = B.copy()
            count = 0
        else:
            count += 1
        if count >= 5:
            break

    return best_B


def getBestHash(cfg, args, train_loader, bit, device, initWithCSQ=False, rho = 5e-5, gamma = (1 + 5 ** 0.5) / 2):

    # PCA Dimensionality Reduction
    features = compute_features(train_loader, cfg,device) 

    im2cluster, Semantic_Center = run_kmeans(features,cfg,args.gpu,device)  #run kmeans clustering on master node
    Semantic_Center = Semantic_Center.cpu().numpy()
    with h5py.File(cfg.semantic_center_path, 'w') as f:
            f.create_dataset('matrix', data=Semantic_Center)

    # placeholder for clustering result
    cluster_result = {'im2cluster':[],'centroids':[]}   

    cluster_result['im2cluster'].append(im2cluster)

    d_max, d_min = get_margin(bit, cfg.nclusters)

    # hash centers initialization
    random.seed(80)
    np.random.seed(80)

    d = bit - 2 * d_max
    if initWithCSQ:
        Hash_Center = CSQ_init(cfg.nclusters, bit)  # initialize with CSQ
    else:
        Hash_Center = init_hash(cfg.nclusters, bit)  # random initialization

    # metric initialization
    best_st = eval_metrics(Hash_Center,Semantic_Center)

    best_Hash_Center = copy.deepcopy(Hash_Center)

    best_st = -999999

    # Update Hash Center
    best_Hash_Center = Lp_box(Hash_Center, best_Hash_Center, cfg.nclusters, d_max, bit, rho, gamma, best_st, Semantic_Center)

    best_Hash_Center = torch.tensor(best_Hash_Center).to(device)
    cluster_result['centroids'].append(best_Hash_Center)
    best_Hash_Center = best_Hash_Center.cpu().numpy()

    # Save best_Hash_Center_numpy Matrix to an HDF5 File
    with h5py.File(cfg.hash_center_path, 'w') as f:
        f.create_dataset('matrix', data=best_Hash_Center)

    return cluster_result


# Calculate the Accuracy of k-means Clustering
def cluster_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    matched_count = cm[row_ind, col_ind].sum()
    accuracy = matched_count / len(y_true)
    return accuracy
