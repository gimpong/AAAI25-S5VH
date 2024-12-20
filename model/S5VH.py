import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math
import warnings
import numpy as np
from functools import partial
import math
from typing import Optional
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm import Mamba

from .RWKV.vrwkv6 import RWKVBLOCK
from .RetNet.retention import MultiScaleRetention
from .Transformer.transformerBlock import TransformerBlock
import h5py

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

	
def __call_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

		
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = ~(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output * mask
        return grad_output, None, None


class DropPath(nn.Module):
    '''
    drop paths (stochastic depth) per sample, (when applied in main path of residual blocks)
    '''
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p = {}'.format(self.drop_prob)


class Block(nn.Module):
    """
    Bidirectional Mamba to Extract Semantic Information
    """
    def __init__(self, dim,  mlp_ratio=4.,drop=0., layer_id=0.,n_layer=12,
                    drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        self.layer_id = layer_id
        self.n_head = dim//64
        self.n_layer = n_layer
        self.init_values = init_values
        self.rwkv_drop_path = drop_path
        self.mamba1 = Mamba(
                d_model=dim,d_state=16,d_conv=4,expand=2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.dim = dim
        self.mamba2 =Mamba(
                d_model=dim,d_state=16,d_conv=4,expand=2
        )
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, model_name="S5VH"):

        if model_name == "LSTM":
            # Replacing Mamba with LSTM
            self.mamba1 = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=1, batch_first=True)
            self.mamba2 = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=1, batch_first=True)
            self.mamba1 = self.mamba1.to(x.device)
            self.mamba2 = self.mamba1.to(x.device)
            x_up, _ = self.mamba1(self.norm1(x))
            flip_x = torch.flip(x,dims=[1])
            x_down, _ = self.mamba2(self.norm2(flip_x))
            x_down = torch.flip(x_down,dims=[1])
            x = x+self.drop_path(x_up+x_down)
            return x
        elif model_name == "Transformer":
            # Replacing Mamba with Tranformer
            x = TransformerBlock(x)
            return x
        elif model_name == "RWKV":
            # Replacing Mamba with RWKV 
            self.mamba1 = RWKVBLOCK(
                n_embd=self.dim, hidden_rate=self.mlp_ratio,n_head=self.n_head,layer_id=self.layer_id,n_layer=self.n_layer,
                drop_path=self.rwkv_drop_path, 
                init_values=self.init_values)
            self.mamba2 = RWKVBLOCK(
                n_embd=self.dim, hidden_rate=self.mlp_ratio,n_head=self.n_head,layer_id=self.layer_id,n_layer=self.n_layer,
                drop_path=self.rwkv_drop_path, 
                init_values=self.init_values)
        elif model_name == "RetNet":
            self.mamba1 = MultiScaleRetention(self.dim, 4, True)
            self.mamba2 = MultiScaleRetention(self.dim, 4, True)
        



        self.mamba1 = self.mamba1.to(x.device)
        self.mamba2 = self.mamba1.to(x.device)
        x_up = self.mamba1(self.norm1(x))
        flip_x = torch.flip(x,dims=[1])
        x_down = torch.flip(self.mamba2(self.norm2(flip_x)),dims=[1])
        x = x+self.drop_path(x_up+x_down)
        return x



def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class S5VHEncoder(nn.Module):
    def __init__(self,feature_num=4096, embed_dim=256, max_frame=25, nbits=64, depth=12, mlp_ratio=4.,
                    drop_rate=0.,drop_path_rate=0., 
                    norm_layer=nn.LayerNorm, init_values=None, use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        num_patches = max_frame
        self.num_patches = num_patches
        self.nbits = nbits
        self.patch_embed = nn.Linear(feature_num, embed_dim)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

        
    def forward_features(self, x, mask, model_name="S5VH"):
        x = self.patch_embed(x)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x=x_vis, model_name = model_name)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask, model_name="S5VH"):
        x = self.forward_features(x, mask, model_name)
        x = self.head(x)
        return x


class S5VHDecoder(nn.Module):
    def __init__(self, feature_num=4096, embed_dim=256, max_frame=25, depth=12,
                  mlp_ratio=4., drop_rate=0., 
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        num_patches = max_frame
        self.num_patches = num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,mlp_ratio=mlp_ratio, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, feature_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, return_token_num, model_name="S5VH"):
        for blk in self.blocks:
            x = blk(x=x, model_name=model_name)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

def readBestHash(cfg):
    # Load hash_center Matrix from an HDF5 File
    with h5py.File(cfg.hash_center_path, 'r') as f:
        hash_center = f['matrix'][:]

    hash_center = torch.tensor(hash_center)

    return hash_center

class S5VH_Model(nn.Module):
    def __init__(self,
                 cfg,
                 feature_num=4096, 
                 encoder_embed_dim=256, 
                 max_frame=25,
                 mask_ratio=0.5,
                 nbits=64,
                 encoder_depth=12,
                 decoder_embed_dim=256, 
                 decoder_depth=8,
                 mlp_ratio=4., 
                 drop_rate=0.5, 
                 drop_path_rate=0.5, 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 ):
        super(S5VH_Model, self).__init__()
        self.cfg = cfg
        self.nclusters = cfg.nclusters
        
        self.visible_patches = int(max_frame*(1-mask_ratio))
        self.get_token_probs = nn.Sequential(
            nn.Linear(feature_num,encoder_embed_dim),
            Mamba(d_model=encoder_embed_dim,d_state=16,d_conv=4,expand=2),
            nn.Linear(encoder_embed_dim,1),
            nn.Flatten(start_dim=1),
        )
        self.softmax = nn.Softmax(dim=-1)
        
        
        self.encoder = S5VHEncoder(
            feature_num=feature_num, 
            embed_dim=encoder_embed_dim, 
            max_frame=max_frame,
            nbits=nbits,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio, 
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = S5VHDecoder(
            feature_num=feature_num, 
            embed_dim=decoder_embed_dim, 
            max_frame=max_frame,
            depth=decoder_depth,
            mlp_ratio=mlp_ratio, 
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.encoder_to_decoder = nn.Linear(self.encoder.nbits, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim)

        self.binary = nn.Linear(self.encoder.num_features, self.encoder.nbits)
        self.ln = nn.LayerNorm(self.encoder.nbits)
        self.activation = self.binary_tanh_unit
        
        self.classifier = nn.Linear(nbits,self.nclusters,bias = False)
        hash_center = readBestHash(cfg)
        self.classifier.data = hash_center
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def binary_tanh_unit(self, x):
        y = self.hard_sigmoid(x)
        out = 2. * Round3.apply(y) - 1.
        return out

    def hard_sigmoid(self, x):
        y = (x + 1.) / 2.
        y[y > 1] = 1
        y[y < 0] = 0
        return y

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    @torch.no_grad()
    def get_label_pred(self, x):
        """
            x: fullview
            Pass in fullview to obtain the model's prediction vector.
        """
        batch_size = x.size(0)
        frame_num = x.size(1)
        device = x.device
        mask = torch.zeros((batch_size, frame_num), dtype=torch.bool, device=device)
        x_feat = self.encoder(x, mask, model_name=self.cfg.model_name)
        hash_code = self.binary(x_feat)
        hash_code = self.activation(hash_code)
        cluster_preds  = self.classifier(torch.mean(hash_code,1))

        return cluster_preds

    def forward(self,x,mask):

        x_feat = self.encoder(x, mask, model_name=self.cfg.model_name)
        hash_code = self.binary(x_feat)
        hash_code = self.ln(hash_code)
        hash_code = self.activation(hash_code)
        device = x.device

        cluster_preds = self.classifier(torch.mean(hash_code,1))
        
        x_vis = self.encoder_to_decoder(hash_code)
        B, N, C = x_vis.shape
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        output = self.decoder(x_full, pos_emd_mask.shape[1], model_name=self.cfg.model_name)

        return output, hash_code,cluster_preds

    def inference(self, x, model_name):
        batch_size = x.size(0)
        frame_num = x.size(1)
        device = x.device
        mask = torch.zeros((batch_size, frame_num), dtype=torch.bool, device=device)

        x = self.encoder(x, mask,model_name)

        x = self.binary(x)
        x = self.ln(x)
        x = self.activation(x)
        return x
    
def S5VH(cfg):
    # Choose models of different sizes.
    assert cfg.S5VH_type in ['mini', 'small', 'base']

    if cfg.S5VH_type == 'mini':
        model = S5VH_Model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            mask_ratio=cfg.mask_ratio,
            nbits=cfg.nbits,
            encoder_depth=1,
            decoder_embed_dim=192,
            decoder_depth=1,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        return model
    elif cfg.S5VH_type == 'small':
        model = S5VH_Model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            mask_ratio=cfg.mask_ratio,
            nbits=cfg.nbits,
            encoder_depth=6,
            decoder_embed_dim=192,
            decoder_depth=1,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        return model
    elif cfg.S5VH_type == 'base':
        model = S5VH_Model(
            cfg=cfg,
            feature_num=cfg.feature_size, 
            encoder_embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            mask_ratio=cfg.mask_ratio,
            nbits=cfg.nbits,
            encoder_depth=12,
            decoder_embed_dim=192,
            decoder_depth=1,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        return model
    