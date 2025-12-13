# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

def get_len_mask(features_lens):
    features_lens = (features_lens + 1) // 2
    features_lens = (features_lens + 1) // 2
    batch_size = len(features_lens)
    max_len = torch.max(features_lens)
    mask = torch.zeros((batch_size,max_len),dtype=torch.float32)
    for i in range(batch_size):
        mask[i,0:features_lens[i]] = 1.0
    return mask

def restore(x1,x2):
    x1 = x1.transpose(0,1)
    x2 = x2.transpose(0,1)
    x1_len = x1.shape[0]
    x2_len = x2.shape[0]
    min_len = min(x1_len,x2_len)
    y = []
    for i in range(min_len):
        y.append(x1[i].unsqueeze(0))
        y.append(x2[i].unsqueeze(0))
    if x2_len < x1_len:
        y.append(x1[-1].unsqueeze(0))
    y = torch.cat(y,dim=0).transpose(0,1)
    return y

class ArbitraryInteractor(nn.Module):
    def __init__(self,d_in,d_out,kernel_size,padding,bias,l):
        super().__init__()
        self.conv1 = nn.Conv1d(d_in,d_out,kernel_size=kernel_size,padding=padding,stride=1,groups=d_in,bias=bias)
        self.conv2 = nn.Conv1d(d_in,d_out,kernel_size=1,bias=bias)
        self.complex_weight = nn.Parameter(torch.randn(d_in,l,2,dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight,std=.02)
        self.head = nn.Linear(d_out,d_out,bias=bias)

    def forward(self,x):
        x1 = x[:,:,0::2]
        x2 = x[:,:,1::2]
        x1 = self.conv1(x1)
        _,_,l = x2.shape
        x2 = torch.fft.rfft(x2,dim=2,norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:2] == x2.shape[2:3]:
            weight = F.interpolate(weight.permute(2,0,1).unsqueeze(2),size=(1,x2.shape[2]),mode='bilinear',align_corners=True).squeeze().permute(1,2,0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 *= weight
        x2 = torch.fft.irfft(x2,n=l,dim=2,norm='ortho')
        x2 = self.conv2(x2)
        y = restore(x1.transpose(1,2),x2.transpose(1,2))
        y = self.head(y)
        return y
    
class MultiscaleInteractor(nn.Module):
    def __init__(self, d_in, d_out, ls=[500,1000,1500]):
        super().__init__()
        self.processors = nn.ModuleList([
            ArbitraryInteractor(d_in, d_out, kernel_size=3, padding=1, bias=True, l=l)
            for l in ls
        ])
        self.attention = nn.MultiheadAttention(d_out, 4)
        
    def forward(self, x):
        multi_scale_features = [p(x) for p in self.processors]
        combined = torch.stack(multi_scale_features, dim=1).mean(dim=1)
        attn_output, _ = self.attention(combined, combined, combined)
        return attn_output

class SelfGatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, temporal, frequency):
        gate = self.gate(torch.cat([temporal, frequency], dim=-1))
        return gate * temporal + (1 - gate) * frequency

class SelectivePool1d(nn.Module):
    def __init__(self,in_features,d_head,num_heads,tau=1.0):
        super().__init__()
        self.keys = nn.Parameter(torch.Tensor(num_heads,d_head),requires_grad=True)
        self.W_q = nn.Conv1d(in_features,d_head * num_heads,kernel_size=1)
        self.norm = 1 / np.sqrt(d_head)
        self.d_head = d_head
        self.num_heads = num_heads
        self.cnt = 0
        self.weights_init()

    def weights_init(self):
        nn.init.orthogonal_(self.keys,gain=1)
        nn.init.kaiming_normal_(self.W_q.weight,a=1)
        nn.init.zeros_(self.W_q.bias)

    def orthogonal_norm(self):
        keys = F.normalize(self.keys,dim=1)
        corr = torch.mm(keys,keys.transpose(0,1))
        return torch.sum(torch.triu(corr,1).abs_())

    def forward(self,x,mask):
        N,_,L = x.shape # (N,C,L)
        q = v = self.W_q(x).transpose(1,2).view(N,L,self.num_heads,self.d_head)
        if mask != None:
            mask = mask.to(x.device)
            attn = F.softmax(torch.sum(q * self.keys,dim=-1) * self.norm - (1. - mask).unsqueeze(2) * 1000,dim=1) # (N,L,num_heads)
        else:
            attn = F.softmax(torch.sum(q * self.keys,dim=-1) * self.norm,dim=1)
        y = torch.sum(v * attn.unsqueeze(3),dim=1).view(N,-1) # (N,d_head * num_heads)
        return y
    
class Head(nn.Module):
    def __init__(self,d_in,d_out,bias=False):
        super().__init__()
        self.fc1 = nn.Linear(d_in,d_out,bias=bias)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_out,1,bias=bias),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        y = self.fc1(x)
        logit = self.fc2(y)
        return logit

class SPECTRUM(nn.Module):
    def __init__(self,d_in,d_out,d_hidden=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(d_in,d_out,kernel_size=7,padding=3,stride=1,bias=True),
            nn.MaxPool1d(2,2,ceil_mode=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_out,d_hidden,kernel_size=3,padding=1,stride=1,bias=True),
            nn.MaxPool1d(2,2,ceil_mode=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.freq_processor1 = MultiscaleInteractor(d_out,d_out,ls=[500,1000,1500])
        self.freq_processor2 = MultiscaleInteractor(d_hidden,d_hidden,ls=[300,600,900])
        self.adaptive_fusion1 = SelfGatedFusion(d_out)
        self.adaptive_fusion2 = SelfGatedFusion(d_hidden)
        self.rnn = nn.GRU(d_hidden,d_hidden,num_layers=2,dropout=0.1,batch_first=True,bidirectional=False)
        for i in range(2):
            eval(f'self.rnn.bias_hh_l{i}')[d_hidden:d_hidden * 2].data.fill_(-1e10)
            eval(f'self.rnn.bias_ih_l{i}')[d_hidden:d_hidden * 2].data.fill_(-1e10)
        self.head = nn.Linear(d_hidden,d_out,bias=False)
        self.pool = SelectivePool1d(d_hidden,d_head=16,num_heads=8)
        self.binary = nn.Sequential(
            nn.Linear(128,64,bias=False),
            nn.Dropout(0.1),
            nn.Linear(64,1,bias=False),
            nn.Sigmoid()
        )
        self.weights_init()
    
    def weights_init(self):
        nn.init.kaiming_normal_(self.conv1[0].weight,a=0)
        nn.init.kaiming_normal_(self.conv2[0].weight,a=0)
        nn.init.zeros_(self.conv1[0].bias)
        nn.init.zeros_(self.conv2[0].bias)
        nn.init.kaiming_normal_(self.head.weight,a=1)
        nn.init.kaiming_normal_(self.binary[0].weight,a=1)
        nn.init.kaiming_normal_(self.binary[-2].weight,a=1)

    def forward(self,x,mask):
        y1 = self.conv1(x.transpose(1,2))
        freq1 = self.freq_processor1(y1)
        fused1 = self.adaptive_fusion1(y1.transpose(1,2),freq1)
        y2 = self.conv2(fused1.transpose(1,2))
        freq2 = self.freq_processor2(y2)
        fused2 = self.adaptive_fusion2(y2.transpose(1,2),freq2)
        fused2 = fused2 * mask.unsqueeze(2)

        features_lens = torch.sum(mask,dim=1)
        y = pack_padded_sequence(fused2,features_lens.cpu(),batch_first=True,enforce_sorted=False)
        y,_ = self.rnn(y)
        y,features_lens = pad_packed_sequence(y,batch_first=True)
        y = self.head(y)
        freq_vec = self.pool(freq2.transpose(1,2),None)
        freq_logit = self.binary(freq_vec) if self.training else None
        y = y * mask.unsqueeze(2)
        return y,features_lens,freq_logit,freq_vec