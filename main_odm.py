#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from utils import *
from dataset import *
from models import *
import numpy as np
import open3d as o3d
from losses import *
from matplotlib import pyplot as plt
from clip_util_odm import test_one_epoch_seq, train_one_epoch_seq
from utils.vis_util import *
from tensorboardX import SummaryWriter as Logger

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'loss_train'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'loss_train')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp configs.yaml checkpoints' + '/' + args.exp_name + 'configs.yaml.backup')


def test(args, net, test_loader, textio):

    sf_metric, seg_metric, pose_metric, gt_trans, pre_trans, pose_metric_kitti = test_one_epoch_seq(args, net, test_loader, textio)

    ## print scene flow evaluation results
    for metric in sf_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, sf_metric[metric]))

    ## print ego_motion evaluation results
    for metric in pose_metric:
        textio.cprint('###The mean {}: {}###'.format(metric, pose_metric[metric]))

    eval_trmse = pose_metric_kitti['TRMSE']
    eval_rrmse = pose_metric_kitti['RRMSE']
    textio.cprint('TRMSE score: %f' % eval_trmse)
    textio.cprint('RRMSE score: %f' % eval_rrmse)

    textio.cprint('Max memory alocation: {}MB'.format(torch.cuda.max_memory_allocated(device=0)/1e6)) 
    print('FINISH')
  
def train(args, net, train_loader, val_loader, test_loader, textio, tb_logger):
    
    
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(opt, args.decay_epochs, gamma = args.decay_rate)

    best_val_res = np.inf
    train_loss_ls = np.zeros((args.epochs))
    val_score_ls = np.zeros(args.epochs)
    train_items_iter = {
                    'Loss': [],'nnLoss': [],'smoothnessLoss': [],'veloLoss': [],
                    'cycleLoss': [],'curvatureLoss':[],'chamferLoss': [],'L2Loss': [], 'glLoss': [],
                    'egoLoss':[], 'maskLoss': [], 'superviseLoss': [], 'opticalLoss': [], 'L1Loss': [],
                    'odmLoss': [], 'tLoss': [], 'qLoss': [],
                    }
    
    for epoch in range(args.epochs):
        
        textio.cprint('====epoch: %d, learning rate: %f===='%(epoch, opt.param_groups[0]['lr']))

        textio.cprint('==starting training on the training set==')
        if args.dataset == 'vodClipDataset' or args.dataset == 'vodClipDatasetOdm':
            total_loss, loss_items = train_one_epoch_seq(args, net, train_loader, opt)
        else:
            total_loss, loss_items = train_one_epoch(args, net, train_loader, opt)

        train_loss_ls[epoch] = total_loss
        for it in loss_items:
            train_items_iter[it].extend([loss_items[it]])
        textio.cprint('mean train loss: %f'%total_loss)
        tb_logger.add_scalar('total_loss', total_loss, epoch)
        tb_logger.add_scalar('ego_loss', loss_items['odmLoss'], epoch)
        tb_logger.add_scalar('tLoss', loss_items['tLoss'], epoch)
        tb_logger.add_scalar('qLoss', loss_items['qLoss'], epoch)

        textio.cprint('==starting evaluation on the validation set==')
        if args.dataset == 'vodClipDataset' or args.dataset == 'vodClipDatasetOdm':
            sf_metric, _, pose_metric, _, _, pose_metric_kitti = test_one_epoch_seq(args, net, test_loader, textio)
        else:
            sf_metric, _, pose_metric, _, _, pose_metric_kitti = eval_one_epoch(args, net, test_loader, textio)
        eval_score = sf_metric['rne']
        val_score_ls[epoch] = eval_score
        textio.cprint('mean RNE score: %f'%eval_score)
        ###
        eval_rte = pose_metric['RTE']
        eval_rae = pose_metric['RAE']
        textio.cprint('RTE score: %f' % eval_rte)
        textio.cprint('RAE score: %f' % eval_rae)
        tb_logger.add_scalar('RTE', eval_rte, epoch)
        tb_logger.add_scalar('RAE', eval_rae, epoch)
        ###
        eval_trmse = pose_metric_kitti['TRMSE']
        eval_rrmse = pose_metric_kitti['RRMSE']
        textio.cprint('TRMSE score: %f' % eval_trmse)
        textio.cprint('RRMSE score: %f' % eval_rrmse)
        tb_logger.add_scalar('TRMSE', eval_trmse, epoch)
        tb_logger.add_scalar('RRMSE', eval_rrmse, epoch)

        if best_val_res >= (eval_trmse + eval_rrmse * 100) / 2:
            best_val_res = (eval_trmse + eval_rrmse * 100) / 2
            textio.cprint('best val TRMSE till now: %f' % eval_trmse)
            textio.cprint('best val RRMSE till now: %f' % eval_rrmse)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % (args.exp_name))
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % (args.exp_name))

        scheduler.step()

    textio.cprint('====best RNE score after %d epochs: %f===='%(args.epochs, best_val_res)) 
    plt.clf()
    plt.plot(train_loss_ls[0:int(args.epochs)], 'b')
    plt.legend(['train_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train/train_loss.png' % args.exp_name,dpi=500)

    plt.clf()
    plt.plot(val_score_ls[0:int(args.epochs)], 'r')
    plt.legend(['val_score'])
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.savefig('checkpoints/%s/val_score.png' % args.exp_name,dpi=500)

    return best_val_res
   

def main(io_args):
    
    args = parse_args_from_yaml("configs.yaml")
    args.scan_map = io_args.scan_map
    args.eval = io_args.eval
    args.vis = io_args.vis
    args.dataset_path = io_args.dataset_path
    args.exp_name = io_args.exp_name
    args.model = io_args.model
    args.save_res = io_args.save_res
    args.dataset = io_args.dataset

    # CUDA settings
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    # deterministic results
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    
    # init checkpoint records 
    _init_(args)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    # init tensorboard
    tb_logger = Logger('checkpoints/' + args.exp_name + "/train_tb")

    # init dataset and dataloader
    if args.eval:
        test_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.eval_split,textio=textio)
        test_loader = DataLoader(test_set,num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
    else:
        train_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition=args.train_set,textio=textio)
        val_set = dataset_dict[args.dataset](args=args, root = args.dataset_path, partition='val', textio=textio)
        args.eval = True  # for cmflow_t
        test_set = dataset_dict[args.dataset](args=args, root=args.dataset_path, partition=args.eval_split, textio=textio)
        args.eval = False  # for cmflow_t
        train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=args.val_batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=1, shuffle=False, drop_last=False)
            
    # update dataset extrisic and intrinsic to args
    if not args.eval:
        args.camera_projection_matrix = train_set.camera_projection_matrix
        args.t_camera_radar = train_set.t_camera_radar
        args.radar_res = train_set.res
    else:
        args.camera_projection_matrix = test_set.camera_projection_matrix
        args.t_camera_radar = test_set.t_camera_radar
        args.radar_res = test_set.res
    if args.eval:
        args.clips_info = test_set.clips_info
    
    # init the network (load or from scratch)
    net = init_model(args)
    
    if args.eval:
        best_val_res = None
        test(args, net, test_loader,textio)
    else:
        best_val_res = train(args, net, train_loader, val_loader, test_loader, textio, tb_logger)

    textio.cprint('Max memory alocation: {}MB'.format(torch.cuda.max_memory_allocated(device=0)/1e6)) 
    print('FINISH')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radar Scene flow running')
    parser.add_argument('--scan_map', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action = 'store_true')
    parser.add_argument('--save_res', action='store_true')
    parser.add_argument('--dataset_path', type= str, default = '/mnt/12T/fangqiang/preprocess_res/flow_smp/')
    parser.add_argument('--exp_name', type = str, default = 'cmflow_cvpr')
    parser.add_argument('--model', type = str, default = 'cmflow')
    parser.add_argument('--dataset', type = str, default = 'vodDataset')
    args = parser.parse_args()
    main(args)