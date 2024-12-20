# model
model_name = 'RWKV'
use_checkpoint = None
feature_size = 4096
hidden_size = 256
max_frames = 25
nbits = 16
S5VH_type = 'small'

# dataset
dataset = 'fcv'
workers = 12
batch_size = 512
mask_ratio = 0.3

# preprocess
latent_dim_pca = 512
initWithCSQ = True
rho = 5e-5
gamma = (1 + 5 ** 0.5) / 2

# train
seed = 1
num_epochs = 500
alpha = 2.5
temperature = 0.5
tau_plus = 0.1
train_num_sample = 45585

# Reliability-Aware Hash Center Alignment
a_cluster = 0.01
temperature_cluster = 0.5
nclusters  = 450
warm_up_epoch = 50
smoothing_alpha = 0.01 

# test
test_batch_size = 128
test_num_sample = 45600

# optimizer
optimizer_name = 'AdamW'
schedule ="CosineAnnealingLR" #'StepLR'
lr = 5e-4
min_lr = 1e-5

# path
data_root = f"data/{dataset}/"
home_root = './'

# path:train
train_feat_path = data_root + 'fcv_train_feats.h5'
train_assist_path = data_root+'final_train_train_assit.h5' 
latent_feat_path = data_root+'final_train_latent_feats.h5'
anchor_path = data_root+'final_train_anchors.h5'
sim_path = data_root+'final_train_sim_matrix.h5'
semantic_center_path = data_root+'semantic.h5'
hash_center_path = f"{data_root}hash_center_{nbits}.h5"


# path:test
test_feat_path = [data_root + 'fcv_test_feats.h5'] # database+query
label_path = [data_root + 'fcv_test_labels.mat']

# path:save
save_dir = home_root + "checkpoint" + dataset
file_path = f"{save_dir}{model_name}_{nbits}bit"
log_path = f"{home_root}logs/{dataset}S5VH_{nbits}bit"