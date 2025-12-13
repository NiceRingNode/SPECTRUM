# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F
from soft_dtw_cuda import SoftDTW

soft_dtw = SoftDTW(True,gamma=5,normalize=False,bandwidth=0.1)

def vanilla_triplet_loss(y_pred,genuine_sample,forgery_sample,features_lens=None,margin=1):
    batch_size = y_pred.size(0)
    sample_interval = genuine_sample + forgery_sample
    user_cnt = batch_size // sample_interval
    outra_loss = 0
    intra_loss = 0
    for idx in range(user_cnt):
        genuine_indices = [i + sample_interval * idx for i in range(0,genuine_sample)]
        anchor_idx = np.random.choice(genuine_indices,size=1,replace=False)[0]
        genuine_indices.remove(anchor_idx)
        forgery_indices = [i + sample_interval * idx for i in range(genuine_sample,sample_interval)]

        anchor = y_pred[anchor_idx]
        forgery = y_pred[forgery_indices]
        genuine = y_pred[genuine_indices]

        len_anchor = features_lens[anchor_idx]
        len_genuine = features_lens[genuine_indices]
        len_forgery = features_lens[forgery_indices]
        dist_ag = torch.zeros((len(genuine))).to(y_pred.device)
        dist_asf = torch.zeros((len(forgery))).to(y_pred.device)
        for i in range(len(genuine)):
            dist_ag[i] = soft_dtw(anchor[None,:int(len_anchor)],genuine[i:i + 1,:int(len_genuine[i])]) / (len_anchor + len_genuine[i])
        for i in range(len(forgery)):
            dist_asf[i] = soft_dtw(anchor[None,:int(len_anchor)],forgery[i:i + 1,:int(len_forgery[i])]) / (len_anchor + len_forgery[i])

        less_than_margin = dist_ag.unsqueeze(1) - dist_asf.unsqueeze(0)
        simple_loss = F.relu(less_than_margin + margin)
        outra_loss += torch.sum(simple_loss) / (simple_loss.data.nonzero(as_tuple=False).size(0) + 1)
        intra_loss += torch.sum(dist_ag) / len(genuine)
    intra_loss /= user_cnt
    outra_loss /= user_cnt
    return intra_loss,outra_loss