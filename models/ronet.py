import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *
from utils import *
from timm.models.layers import DropPath,trunc_normal_
import math
from mamba_ssm import Mamba
LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_activation=True,
                 use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if use_activation:
            relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        else:
            relu = nn.Identity()

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x

class SubFold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x, c):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = c.to(x.device) # b 3 n2
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2

class GeoCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0.,
                 aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.q_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        # self.k_map = nn.Identity()  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)  # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.x_map = nn.Identity()  # nn.Linear(aggregate_dim, 1)

    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)

        x = self.x_map(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_pred=64, num_point=128):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        self.mlp3_convs = nn.ModuleList()
        self.mlp3_bns = nn.ModuleList()

        self.queryandgroup = pointutils.QueryAndGroup(radius=4.0, nsample=8)
        # self.norm_q = nn.Identity()  # norm_layer(dim)
        # self.norm_k = nn.Identity()  # norm_layer(dim)
        self.norm_q = norm_layer(dim)  # norm_layer(dim)
        self.norm_k = norm_layer(dim)  # norm_layer(dim)
        self.attn = GeoCrossAttention(dim, dim, num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop, aggregate_dim=16)

        self.fold_step = int(pow(num_pred, 0.5) + 0.5)
        self.generate_anchor = SubFold(dim, step=self.fold_step, hidden_dim=dim // 2)

    def forward(self, x, coor):
        B, _, _ = x.size()
        idx = pointutils.furthest_point_sample(coor.contiguous(), 64).unsqueeze(-1).to(torch.int64)
        sample_coor = torch.gather(coor, 1, idx.repeat(1, 1, 3))
        sample_x = torch.gather(x, 1, idx.repeat(1, 1, 256))

        local_x = self.queryandgroup(coor.contiguous(), sample_coor.contiguous(), x.permute(0, 2, 1).contiguous())
        local_coor = local_x[:, :3, :, :]
        local_x = local_x[:, 3:, :, :]

        global_x = torch.max(local_x, -1)[0]
        diff_x = (global_x - sample_x.permute(0, 2, 1)).unsqueeze(2)
        diff_coor = torch.mean(local_coor, -1)
        x_2 = diff_x.squeeze(2).permute(0, 2, 1)

        norm_k = self.norm_k(sample_x)  # B N dim
        norm_q = self.norm_q(x_2)  # B L dim
        coor_2 = self.attn(q=norm_q, k=norm_k, v=diff_coor)

        sample_coor = sample_coor + coor_2
        sample_x = sample_x + x_2
        return sample_x, sample_coor

class CAO_RONet(nn.Module):
    
    def __init__(self,args):
        
        super(CAO_RONet,self).__init__()
        
    
        self.npoints = args.num_points
        self.stat_thres = 0.50 #0.50
        
        ## multi-scale set feature abstraction 
        sa_radius = [2.0, 4.0, 8.0, 16.0]
        sa_nsamples = [4, 8, 16, 32]
        sa_mlps = [32, 32, 64]
        sa_mlp2s = [64, 64, 64]
        num_sas = len(sa_radius)
        self.mse_layer = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=3, \
                                         mlp = sa_mlps, mlp2 = sa_mlp2s)

        ## feature correlation layer (cost volumn)
        fc_inch = num_sas*sa_mlp2s[-1]*2  
        fc_mlps = [fc_inch,fc_inch,fc_inch]
        self.fc_layer = FeatureCorrelator(8, in_channel=(fc_inch+3)*3, mlp=fc_mlps)
        
        ## multi-scale set feature abstraction 
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse_layer2 = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)

        self.mp_q = nn.Sequential(
            Conv1d(512, 256, use_activation=True),
            Conv1d(256, 128, use_activation=True),
            Conv1d(128, 4, use_activation=False)
            )
        self.mp_t = nn.Sequential(
            Conv1d(512, 256, use_activation=True),
            Conv1d(256, 128, use_activation=True),
            Conv1d(128, 3, use_activation=False)
            )

        self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)

        self.a_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.a_q = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.encoder = EncoderBlock(
                dim=256, num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., num_pred=64, num_point=256+64)

        self.layer_num = 5
        self.attn = nn.ModuleList()
        for _ in range(self.layer_num):
            self.attn.append(Block_mamba(dim=256, mlp_ratio=4))

    def rigid_to_flow(self,pc,trans):
        
        h_pc = torch.cat((pc,torch.ones((pc.size()[0],1,pc.size()[2])).cuda()),dim=1)
        sf = torch.matmul(trans,h_pc)[:,:3] - pc
        return sf

    def Backbone(self,pc1,pc2,feature1,feature2,gfeat_prev):

        '''
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N

        '''
        B = pc1.size()[0]
        N = pc1.size()[2]

        ## extract multi-scale local features for each point
        pc1_features = self.mse_layer(pc1, feature1)
        pc2_features = self.mse_layer(pc2, feature2)

        pc1_, pc2_ = torch.clone(pc1).permute(0, 2, 1), torch.clone(pc2).permute(0, 2, 1)
        pc1_features_, pc2_features_ = torch.clone(pc1_features).permute(0, 2, 1), torch.clone(pc2_features).permute(0, 2, 1)
        pc1_features_, pc1_ = self.encoder(pc1_features_, pc1_)
        pc2_features_, pc2_ = self.encoder(pc2_features_, pc2_)
        pc1_, pc2_ = pc1_.permute(0, 2, 1), pc2_.permute(0, 2, 1)
        pc1_features_, pc2_features_ = pc1_features_.permute(0, 2, 1), pc2_features_.permute(0, 2, 1)

        pc1 = torch.cat((pc1, pc1_), dim=-1)
        pc2 = torch.cat((pc2, pc2_), dim=-1)
        pc1_features = torch.cat((pc1_features, pc1_features_), dim=-1)
        pc2_features = torch.cat((pc2_features, pc2_features_), dim=-1)

        ## global features for each set
        gfeat_1 = torch.max(pc1_features, -1)[0].unsqueeze(2).expand(pc1_features.size()[0], pc1_features.size()[1],
                                                                     pc1.size()[2])
        gfeat_2 = torch.max(pc2_features, -1)[0].unsqueeze(2).expand(pc2_features.size()[0], pc2_features.size()[1],
                                                                     pc2.size()[2])

        ## concat local and global features
        pc1_features = torch.cat((pc1_features, gfeat_1), dim=1)
        pc2_features = torch.cat((pc2_features, gfeat_2), dim=1)

        ## associate data from two sets
        ################################
        idx1 = pointutils.furthest_point_sample(pc1.permute(0, 2, 1).contiguous(), 64).unsqueeze(-1).to(torch.int64)
        pc1_ = torch.gather(pc1.permute(0, 2, 1), 1, idx1.repeat(1, 1, 3)).permute(0, 2, 1)
        pc1_features_ = torch.gather(pc1_features.permute(0, 2, 1), 1, idx1.repeat(1, 1, 512)).permute(0, 2, 1)

        idx2 = pointutils.furthest_point_sample(pc2.permute(0, 2, 1).contiguous(), 64).unsqueeze(-1).to(torch.int64)
        pc2_ = torch.gather(pc2.permute(0, 2, 1), 1, idx2.repeat(1, 1, 3)).permute(0, 2, 1)
        pc2_features_ = torch.gather(pc2_features.permute(0, 2, 1), 1, idx2.repeat(1, 1, 512)).permute(0, 2, 1)

        cor_features = self.fc_layer(pc1, pc2, pc1_features, pc2_features, pc1_, pc2_, pc1_features_, pc2_features_)

        ## generate embeddings
        pc1 = torch.cat((pc1, pc1_), dim=-1)
        pc1_features = torch.cat((pc1_features, pc1_features_), dim=-1)

        embeddings = torch.cat((pc1_features, cor_features), dim=1)
        prop_features = self.mse_layer2(pc1, embeddings)
        gfeat = torch.max(prop_features,-1)[0]

        if gfeat_prev is None:
            gfeat_new = gfeat
            gfeat_new_re = gfeat_new.unsqueeze(1)
        else:
            gfeat_new = torch.cat((gfeat.unsqueeze(1), gfeat_prev), dim=1)
            for id in range(self.layer_num):
                gfeat_new = self.attn[id](gfeat_new)
            gfeat_new = gfeat_new[:, 0, :]
            gfeat_new_re = torch.cat((gfeat_new.unsqueeze(1), gfeat_prev), dim=1)

        gfeat_new_expand = gfeat_new.unsqueeze(2).expand(prop_features.size()[0], prop_features.size()[1],
                                                         pc1.size()[2])

        ## concat gfeat with local features
        final_features = torch.cat((prop_features, gfeat_new_expand), dim=1)

        return final_features, gfeat_new_re

    def quat2mat(self, q):
        '''
        :param q: Bx4
        :return: R: BX3X3
        '''
        batch_size = q.shape[0]
        w, x, y, z = q[:, 0].unsqueeze(1), q[:, 1].unsqueeze(1), q[:, 2].unsqueeze(1), q[:, 3].unsqueeze(1)
        Nq = torch.sum(q ** 2, dim=1, keepdim=True)
        s = 2.0 / Nq
        wX = w * x * s;
        wY = w * y * s;
        wZ = w * z * s
        xX = x * x * s;
        xY = x * y * s;
        xZ = x * z * s
        yY = y * y * s;
        yZ = y * z * s;
        zZ = z * z * s
        a1 = 1.0 - (yY + zZ);
        a2 = xY - wZ;
        a3 = xZ + wY
        a4 = xY + wZ;
        a5 = 1.0 - (xX + zZ);
        a6 = yZ - wX
        a7 = xZ - wY;
        a8 = yZ + wX;
        a9 = 1.0 - (xX + yY)
        R = torch.cat([a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1).view(batch_size, 3, 3)
        return R

    def forward(self, pc1, pc2, feature1, feature2, label_m, mode, gfeat):

        # extract backbone features
        final_features, gfeat = self.Backbone(pc1,pc2,feature1,feature2, gfeat)

        # avg pooling
        avg_features = torch.mean(final_features, dim=2, keepdim=True).permute(0, 2, 1)
        q_coarse = self.mp_q(avg_features)
        t_coarse = self.mp_t(avg_features)

        q_coarse = q_coarse / (torch.sqrt(torch.sum(q_coarse * q_coarse, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        odm_q = torch.squeeze(q_coarse, dim=1)
        odm_t = torch.squeeze(t_coarse, dim=1)

        TT = None
        if mode == 'test':
            qq = odm_q.cpu()
            tt = odm_t.permute(1, 0).cpu()
            RR = self.quat2mat(qq)[0]
            filler = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0)
            TT = torch.cat((torch.cat((RR, tt), dim=-1), filler), dim=0).unsqueeze(0)


        return odm_q, odm_t, self.w_x, self.w_q, self.a_x, self.a_q, TT, gfeat

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=2, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block_mamba(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)  # MambaBlock(d_model=dim)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x)) + self.drop_path(self.attn(x.flip(1))).flip(1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
