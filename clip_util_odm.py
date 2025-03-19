import os
import argparse
import sys
import copy
import torch
# from time import clock
from tqdm import tqdm
import cv2
import open3d as o3d
import numpy as np
from utils import *
from models import *
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
from losses import *
from utils.vis_util import *
import time
from KITTI_odometry_evaluation_tool.evaluation_delft import kittiOdomEval


def train_one_epoch_seq(args, net, train_loader, opt):

    total_loss = 0
    num_examples = 0
    mode = 'train'
    net.train()
    loss_items =  copy.deepcopy(loss_dict[args.model])
    seq_len = train_loader.dataset.mini_clip_len

    # for id in range(len(train_loader)):
    #     train_loader.dataset.__getitem__(id)

    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        # use sequence data in order
        iter_loss = 0
        iter_items = copy.deepcopy(loss_dict[args.model])
        num_examples += args.batch_size
        for j in range(0,seq_len):
            ## reading data from dataloader and transform their format
            # pc1, pc2, ft1, ft2, gt_trans, flow_label, \
            #     fg_mask, interval, radar_u, radar_v, opt_flow = extract_data_info_clip(data, j)
            pc1, pc2, ft1, ft2, gt_trans = extract_data_info_clip(data, j)
        
            batch_size = pc1.size(0)
    
            vel1 = ft1[:,0]
           
            if args.model == 'ronet':
            
                # forward and loss computation
                if j==0:
                    odm_q, odm_t, w_x, w_q, a_x, a_q, _, gfeat = net(pc1, pc2, ft1, ft2, None, mode, None)
                else: 
                    gfeat = gfeat.detach()
                    odm_q, odm_t, w_x, w_q, a_x, a_q, _, gfeat = net(pc1, pc2, ft1, ft2, None, mode, gfeat)

                loss_obj = RadarOdmLoss()
                loss, loss_q, loss_x, items = loss_obj(gt_trans, odm_q, odm_t, w_x, w_q, a_x, a_q)

            opt.zero_grad() 
            loss.backward()
            opt.step()

            iter_loss += loss
            for k in iter_items:
                iter_items[k].append(items[k])
        
        iter_loss = iter_loss/seq_len
        for l in iter_items:
            loss_items[l].append(np.mean(np.array(iter_items[l])))
        total_loss += iter_loss.item() * batch_size

        
    total_loss=total_loss/num_examples
    for l in loss_items:
        loss_items[l]=np.mean(np.array(loss_items[l]))
    
    return total_loss, loss_items


def extract_data_info_clip(seq_data, idx):

    pc1, pc2, ft1, ft2, trans = seq_data
    # pc1, pc2, ft1, ft2, trans, gt, mask, interval, radar_u, radar_v, opt_flow = seq_data
    pc1 = pc1[:,idx].cuda().transpose(2,1).contiguous()
    pc2 = pc2[:,idx].cuda().transpose(2,1).contiguous()
    ft1 = ft1[:,idx].cuda().transpose(2,1).contiguous()
    ft2 = ft2[:,idx].cuda().transpose(2,1).contiguous()
    trans = trans[:,idx].cuda().float()

    return pc1, pc2, ft1, ft2, trans


def test_one_epoch_seq(args, net, test_loader, textio):

    
    net.eval()
    
    if args.save_res: 
        args.save_res_path ='checkpoints/'+args.exp_name+"/results/"
        num_seq = 0
        clip_info = args.clips_info[num_seq]
        seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
        if not os.path.exists(seq_res_path):
            os.makedirs(seq_res_path)

    num_pcs=0 
    
    sf_metric = {'rne':0, '50-50 rne': 0, 'mov_rne': 0, 'stat_rne': 0,\
                 'sas': 0, 'ras': 0, 'epe': 0, 'accs': 0, 'accr': 0}

    seg_metric = {'acc': 0, 'miou': 0, 'sen': 0}
    pose_metric = {'RTE': 0, 'RAE': 0}

    gt_trans_all = torch.zeros((len(test_loader),4,4)).cuda()
    pre_trans_all = torch.zeros((len(test_loader),4,4)).cuda()

    infer_time_all = []
    with torch.no_grad():
        clips_info = test_loader.dataset.clips_info
        clips_name = []
        clips_st_index = []
        # extract clip info
        for i in range(len(clips_info)):
           clips_name.append(clips_info[i]['clip_name'])
           clips_st_index.append(clips_info[i]['index'][0])
        # read data in order
        num_clip = 0
        seq_len = test_loader.dataset.update_len
        for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
            
            ## reading data from dataloader and transform their format
            # pc1, pc2, ft1, ft2, trans, gt, \
            #     mask, interval, radar_u, radar_v, opt_flow = extract_data_info(data)
            pc1, pc2, ft1, ft2, trans = extract_data_info(data)

            # start point for inference
            torch.cuda.synchronize()
            start_point = time.time()
            if args.model in ['ronet']:
                #if i==clips_st_index[num_clip]:
                if i==clips_st_index[num_clip] or i%seq_len==0:
                    odm_q, odm_t, w_x, w_q, a_x, a_q, pred_t, gfeat = net(pc1, pc2, ft1, ft2, None, 'test', None)
                    if num_clip<(len(clips_name)-1):
                        num_clip +=1
                else:
                    odm_q, odm_t, w_x, w_q, a_x, a_q, pred_t, gfeat = net(pc1, pc2, ft1, ft2, None, 'test', gfeat)

            # end point for inference
            torch.cuda.synchronize()
            infer_time = time.time() - start_point
            infer_time_all.append(infer_time)
            ## use estimated scene to warp point cloud 1 
            # pc1_warp=pc1+pred_f

            if args.save_res:
                res = {
                    'pc1': pc1[0].cpu().numpy().tolist(),
                    'pc2': pc2[0].cpu().numpy().tolist(),
                    'pred_f': pred_f[0].cpu().detach().numpy().tolist(),
                    'pred_m': pred_m[0].cpu().detach().numpy().astype(float).tolist(),
                    'pred_t': pred_t[0].cpu().detach().numpy().astype(float).tolist(),
                }
                
                if num_pcs < clip_info['index'][1]:
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                else:
                    num_seq += 1
                    clip_info = args.clips_info[num_seq]
                    seq_res_path = os.path.join(args.save_res_path, clip_info['clip_name'])
                    if not os.path.exists(seq_res_path):
                        os.makedirs(seq_res_path)
                    res_path = os.path.join(seq_res_path, '{}.json'.format(num_pcs))
                
                ujson.dump(res,open(res_path, "w"))


            if args.vis:
                visulize_result_2D_pre(pc1, pc2, pred_f, pc1_warp, gt, num_pcs, mask, args)
                visulize_result_2D_seg_pre(pc1, pc2, mask, pred_m, num_pcs, args)

            pred_trans = pred_t
            gt_trans_all[num_pcs:num_pcs+1] = trans
            pre_trans_all[num_pcs:num_pcs+1] = pred_trans   

            pose_res = eval_trans_RPE(trans, pred_trans)
            for metric in pose_res:
                pose_metric[metric] += pose_res[metric]
                
            num_pcs+=1

    infer_time = sum(infer_time_all) / len(infer_time_all)
    for metric in sf_metric:
        sf_metric[metric] = sf_metric[metric]/num_pcs
    for metric in seg_metric:
        seg_metric[metric] = seg_metric[metric]/num_pcs
    for metric in pose_metric:
        pose_metric[metric] = pose_metric[metric]/num_pcs

    ##
    for clip_info in test_loader.dataset.clips_info:
        gt_start = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        pred_start = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        gt_pose = [gt_start]
        pred_pose = [pred_start]
        gt_trans_each = gt_trans_all[clip_info['index'][0]: clip_info['index'][1], :, :].cpu().numpy()
        pre_trans_each = pre_trans_all[clip_info['index'][0]: clip_info['index'][1], :, :].cpu().numpy()

        for frame_id in range(gt_trans_each.shape[0]):
            gt_trans = gt_trans_each[frame_id, :, :]
            gt_pose_now = np.dot(gt_pose[frame_id], T_inv_function(gt_trans))
            gt_pose.append(gt_pose_now)

            pred_trans = pre_trans_each[frame_id, :, :]
            pred_pose_now = np.dot(pred_pose[frame_id], T_inv_function(pred_trans))
            pred_pose.append(pred_pose_now)

        gt_pose = np.stack(gt_pose)
        gt_pose = gt_pose.reshape((gt_pose.shape[0], -1))[:, :12]

        pred_pose = np.stack(pred_pose)
        pred_pose = pred_pose.reshape((pred_pose.shape[0], -1))[:, :12]

        np.savetxt('./eval_result/ground_truth_pose_delft/' + clip_info['clip_name'] + '.txt', gt_pose, fmt='%f')
        np.savetxt('./eval_result/data_delft/' + clip_info['clip_name'] + '.txt', pred_pose, fmt='%f')

    textio.cprint('###The inference speed is %.4fms per frame###'%(infer_time*1000/num_pcs))

    pose_eval = kittiOdomEval()
    pose_metric_kitti = pose_eval.eval(toCameraCoord=False)   # set the value according to the predicted results

    return sf_metric, seg_metric, pose_metric, gt_trans_all, pre_trans_all, pose_metric_kitti

def T_inv_function(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = np.linalg.inv(R)
    T_inv[:3, 3] = -np.linalg.inv(R) @ t

    return T_inv

def extract_data_info(data):

    pc1, pc2, ft1, ft2, trans = data
    pc1 = pc1.cuda().transpose(2,1).contiguous()
    pc2 = pc2.cuda().transpose(2,1).contiguous()
    ft1 = ft1.cuda().transpose(2,1).contiguous()
    ft2 = ft2.cuda().transpose(2,1).contiguous()
    trans = trans.cuda().float()

    return pc1, pc2, ft1, ft2, trans