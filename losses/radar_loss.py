import gc
import argparse
import sys
from pandas import interval_range
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from utils import *
from utils.model_utils import *
import torch.nn.functional as F
from torch.nn import Module

class RadarOdmLoss(Module):

    def __init__(self):
        super(RadarOdmLoss, self).__init__()

    def mat2euler(self, M, seq='zyx'):
        r11 = M[0, 0];
        r12 = M[0, 1];
        r13 = M[0, 2]
        r21 = M[1, 0];
        r22 = M[1, 1];
        r23 = M[1, 2]
        r31 = M[2, 0];
        r32 = M[2, 1];
        r33 = M[2, 2]

        cy = torch.sqrt(r33 * r33 + r23 * r23)

        z = torch.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = torch.atan2(r13, cy)  # atan2(sin(y), cy)
        x = torch.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))

        return z, y, x

    def euler2quat(self, z, y, x):
        z = z / 2.0
        y = y / 2.0
        x = x / 2.0
        cz = torch.cos(z)
        sz = torch.sin(z)
        cy = torch.cos(y)
        sy = torch.sin(y)
        cx = torch.cos(x)
        sx = torch.sin(x)
        return torch.tensor([cx * cy * cz - sx * sy * sz,
                             cx * sy * sz + cy * cz * sx,
                             cx * cz * sy - sx * cy * sz,
                             cx * cy * sz + sx * cz * sy]).cuda()

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

    def forward(self, gt_trans, odm_q, odm_t, w_x, w_q, a_x, a_q):
        BS = gt_trans.shape[0]
        odm_q_gt = 0
        odm_t_gt = 0

        for id in range(BS):
            cur_T_gt = gt_trans[id, :, :]

            cur_R_gt = cur_T_gt[:3, :3]
            z_euler, y_euler, x_euler = self.mat2euler(cur_R_gt)
            cur_q_gt = self.euler2quat(z_euler, y_euler, x_euler)
            cur_q_gt = torch.unsqueeze(cur_q_gt, dim=0)
            cur_t_gt = cur_T_gt[:3, 3:]
            cur_t_gt = torch.unsqueeze(cur_t_gt, dim=0)

            if id == 0:
                odm_q_gt = cur_q_gt
                odm_t_gt = cur_t_gt
            else:
                odm_q_gt = torch.cat([odm_q_gt, cur_q_gt], 0)
                odm_t_gt = torch.cat([odm_t_gt, cur_t_gt], 0)


        odm_t_gt = torch.squeeze(odm_t_gt)

        odm_q_norm = odm_q / (torch.sqrt(torch.sum(odm_q * odm_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        loss_q = torch.mean(torch.sqrt(torch.sum((odm_q_gt - odm_q_norm) * (odm_q_gt - odm_q_norm), dim=-1, keepdim=True) + 1e-10))
        loss_x = torch.mean(torch.sqrt((odm_t - odm_t_gt) * (odm_t - odm_t_gt) + 1e-10))
        total_loss = loss_x * torch.exp(-w_x) + w_x + loss_q * torch.exp(-w_q) + w_q
        item = {'odmLoss': total_loss.item(),
                'tLoss': loss_x.item(),
                'qLoss': loss_q.item()}

        return total_loss, loss_q, loss_x, item

       




            





