#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import ujson
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

class vodClipDatasetOdm(Dataset):

    def __init__(self, args, root='/mnt/12T/fangqiang/vod_unanno/flow_smp/', partition='train', textio=None):

        self.npoints = args.num_points
        self.textio = textio
        self.calib_path = 'dataset/vod_radar_calib.txt'
        self.res = {'r_res': 0.2, # m
                    'theta_res': 1.5 * np.pi/180, # radian
                    'phi_res': 1.5 *np.pi/180  # radian
                }
        self.read_calib_files()
        self.eval = args.eval
        self.partition = partition
        self.root = os.path.join(root, self.partition)
        self.interval = 0.10
        self.mini_clip_len = args.mini_clip_len
        self.update_len = args.update_len
        self.clips = sorted(os.listdir(self.root),key=lambda x:int(x.split("_")[1]))
        self.mini_samples = []
        self.samples = []
        self.clips_info = []
        self.mini_clips_info = []
        
        for clip in self.clips:
            clip_path = os.path.join(self.root, clip)
            samples = sorted(os.listdir(clip_path),key=lambda x:int(x.split("/")[-1].split("_")[0]))

            if self.eval:
                self.clips_info.append({'clip_name':clip, 
                                    'index': [len(self.samples),len(self.samples)+len(samples)]
                                })
            
                for j in range(len(samples)):
                     self.samples.append(os.path.join(clip_path, samples[j]))

            if not self.eval: 
                clip_num = int(np.floor(len(samples)/self.mini_clip_len))
                ## take mini_clip as a sample
                for i in range(clip_num):
                    st_idx = i * self.mini_clip_len
                    mini_sample = []
                    for j in range(self.mini_clip_len):
                        mini_sample.append(os.path.join(clip_path, samples[st_idx+j]))
                        self.samples.append(os.path.join(clip_path, samples[st_idx+j]))
                    self.mini_samples.append(mini_sample)

        if not self.eval:
            self.textio.cprint(self.partition + ' : ' +  str(len(self.mini_samples)) + ' mini_clips')
        if self.eval:
            self.textio.cprint(self.partition + ' : ' +  str(len(self.samples)) + ' frames')

    def __getitem__(self, index):
        
        if not self.eval:
            return self.get_clip_item(index)
        if self.eval:
            with open(self.samples[index], 'rb') as fp:
                data = ujson.load(fp)
            return self.get_sample_item(data)


    def get_sample_item(self, data, aug_T_trans=None):

        data_1 = np.array(data["pc1"]).astype('float32')
        data_2 = np.array(data["pc2"]).astype('float32')
            
        # read input data and features
        pos_1 = data_1[:,0:3]
        pos_2 = data_2[:,0:3]
        feature_1 = data_1[:,[4,3,3]]
        feature_2 = data_2[:,[4,3,3]]

        # static points transformation from frame 1 to frame 2  
        trans = np.linalg.inv(np.array(data["trans"])).astype('float32')

        # augment points
        if not self.eval:
            trans, pos_1, pos_2 = aug_trans(aug_T_trans, trans, pos_1, pos_2)

        sample_idx1, sample_idx2 = self.sample_points_fps(pos_1, pos_2)
        pos_1 = pos_1[sample_idx1,:]
        pos_2 = pos_2[sample_idx2,:]
        feature_1 = feature_1[sample_idx1, :]
        feature_2 = feature_2[sample_idx2, :]

        return pos_1, pos_2, feature_1, feature_2, trans,


    def get_clip_item(self, index):

        mini_sample = self.mini_samples[index]
        mini_pos_1 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32')
        mini_pos_2 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32')
        mini_feat_1 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32')
        mini_feat_2 = np.zeros((self.mini_clip_len, self.npoints,3)).astype('float32') 
        mini_trans = np.zeros((self.mini_clip_len, 4, 4)).astype('float32')

        # argumentation
        aug_T_trans = aug_matrix()

        for i in range(0,len(mini_sample)):
            with open(mini_sample[i], 'rb') as fp:
                data = ujson.load(fp)

            pos_1, pos_2, feature_1, feature_2,trans = self.get_sample_item(data, aug_T_trans)

            # accumulate sample information
            mini_pos_1[i] = pos_1
            mini_pos_2[i] = pos_2
            mini_feat_1[i] = feature_1
            mini_feat_2[i] = feature_2
            mini_trans[i] = trans

        return mini_pos_1, mini_pos_2, mini_feat_1, mini_feat_2, mini_trans

    def read_calib_files(self):
        with open(self.calib_path, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
        self.camera_projection_matrix = intrinsic
        self.t_camera_radar = extrinsic

    def sample_points(self, npts_1, npts_2,):

        if npts_1<self.npoints:
            sample_idx1 = np.arange(0,npts_1)
            sample_idx1 = np.append(sample_idx1, np.random.choice(npts_1,self.npoints-npts_1,replace=True))
        else:
            sample_idx1 = np.random.choice(npts_1, self.npoints, replace=False)
        if npts_2<self.npoints:
            sample_idx2 = np.arange(0,npts_2)
            sample_idx2 = np.append(sample_idx2, np.random.choice(npts_2,self.npoints-npts_2,replace=True))
        else:
            sample_idx2 = np.random.choice(npts_2, self.npoints, replace=False)   
        return sample_idx1, sample_idx2

    def sample_points_fps(self, ps_1, ps_2):
        npts_1, npts_2 = ps_1.shape[0], ps_2.shape[0]

        if npts_1<self.npoints:
            sample_idx1 = np.arange(0,npts_1)
            padding_idx = farthest_point_sample(ps_1, self.npoints-npts_1)
            sample_idx1 = np.append(sample_idx1, padding_idx)
        else:
            sample_idx1 = farthest_point_sample(ps_1, self.npoints)

        if npts_2<self.npoints:
            sample_idx2 = np.arange(0,npts_2)
            padding_idx = farthest_point_sample(ps_2, self.npoints-npts_2)
            sample_idx2 = np.append(sample_idx2, padding_idx)
        else:
            sample_idx2 = farthest_point_sample(ps_2, self.npoints)

        return sample_idx1, sample_idx2

    def __len__(self):

        if not self.eval:
            return len(self.mini_samples)
        if self.eval:
            return len(self.samples)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)

def aug_matrix():
    anglex = 0
    angley = 0
    anglez = 0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)

    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])

    scale = np.diag(np.random.uniform(1.00, 1.00, 3).astype(np.float32))
    R_trans = Rx.dot(Ry).dot(Rz).dot(scale.T)

    xx = np.clip(0.01 * np.random.randn(), -1.0, 1.0).astype(np.float32)
    yy = np.clip(0.01 * np.random.randn(), -1.0, 1.0).astype(np.float32)
    zz = np.clip(0.01 * np.random.randn(), -1.0, 1.0).astype(np.float32)

    add_xyz = np.array([[xx], [yy], [zz]])

    T_trans = np.concatenate([R_trans, add_xyz], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)
    T_trans = np.concatenate([T_trans, filler], axis=0)

    return T_trans

def aug_trans(aug_T_trans, trans, pos_1, pos_2):
    aug_frame = np.random.choice([1, 2], replace=True)  # random choose aug frame 1 or 2
    aug_T_trans_inv = np.linalg.inv(aug_T_trans)

    if aug_frame == 2:
        padding = np.zeros((pos_2.shape[0], 1))
        pos_2_aug = np.concatenate((pos_2, padding), axis=1)

        pos_2_aug = np.transpose(pos_2_aug)
        pos_2_aug = np.dot(aug_T_trans, pos_2_aug)
        pos_2 = np.transpose(pos_2_aug)

        trans = np.dot(aug_T_trans, trans)

    elif aug_frame == 1:
        padding = np.zeros((pos_1.shape[0], 1))
        pos_1_aug = np.concatenate((pos_1, padding), axis=1)

        pos_1_aug = np.transpose(pos_1_aug)
        pos_1_aug = np.dot(aug_T_trans, pos_1_aug)
        pos_1 = np.transpose(pos_1_aug)

        trans = np.dot(trans, aug_T_trans_inv)

    return trans, pos_1[:, :3], pos_2[:, :3]

def T_inv_function(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = np.linalg.inv(R)
    T_inv[:3, 3] = -np.linalg.inv(R) @ t

    return T_inv