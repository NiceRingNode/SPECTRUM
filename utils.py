# -*- coding:utf-8 -*-

import numpy as np
import logging,sys,time,os,scipy
from functools import lru_cache
from termcolor import colored
import torch
import torch.nn.functional as F

class ButterWorthLPF:
    def __init__(self,order=3,half_pnt=10.0,fnyquist=200.0):
        fM = 0.5 * fnyquist
        half_pnt /= fM
        b,a = scipy.signal.butter(order,half_pnt,'low')
        self.b = b
        self.a = a

    def __call__(self,x):
        return scipy.signal.filtfilt(self.b,self.a,x)
    
lpf = ButterWorthLPF(half_pnt=15,fnyquist=100)

def difference(x):
    delta_x = np.convolve(x,[1,-1],mode='same')
    delta_x[0] = delta_x[1] / 2
    return delta_x

def line_velocity(x,y):
    # x1 = lpf(x)
    # y1 = lpf(y)
    delta_x = difference(x)
    delta_y = difference(y)
    v = np.sqrt(np.power(delta_x,2) + np.power(delta_y,2))
    return v

def line_acceleration(x,y):
    v = line_velocity(x,y)
    delta_v = difference(v)
    return delta_v

def accumulated_line_velocity(x,y):
    v = line_velocity(x,y)
    # print('accumulated_line_velocity:',np.sum(v))
    y = np.cumsum(np.abs(v))
    return y

def accumulated_line_acceleration(x,y):
    a = line_acceleration(x,y)
    # print('accumulated_line_acceleration:',np.sum(a))
    y = np.cumsum(np.abs(a))
    return y

def calc_angles(x,y):
    # x1 = lpf(x)
    # y1 = lpf(y)
    delta_x = difference(x)
    delta_y = difference(y)
    angles = np.arctan2(delta_y,delta_x)
    return angles

def angular_velocity(x,y):
    angles = calc_angles(x,y)
    delta_angles = difference(angles)
    t = np.where(np.abs(delta_angles) > np.pi)
    delta_angles[t] = np.sign(delta_angles[t]) * 2 * np.pi
    delta_angles *= 0.5
    return delta_angles

def accumulated_angular_velocity(x,y):
    angle_v = angular_velocity(x,y)
    # print('accumulated_angular_velocity:',np.sum(angle_v))
    y = np.cumsum(np.abs(angle_v))
    return y

def angular_acceleration(x,y):
    angle_v = angular_velocity(x,y)
    delta_angle_v = difference(angle_v)
    return delta_angle_v

def accumulated_angular_acceleration(x,y):
    # angle_v = angular_velocity(x,y)
    angle_a = angular_acceleration(x,y)
    # print('accumulated_angular_acceleration:',np.sum(angle_a))
    y = np.cumsum(np.abs(angle_a))
    return y

def freq_magnitude(x):
    v_fft = np.fft.fft(x)
    magnitute = np.abs(v_fft)
    magnitute = np.fft.fftshift(magnitute)
    return magnitute

def difference_theta(x): # 输入的x是角度
    delta_x = np.zeros_like(x)
    delta_x[1:-1] = x[2:] - x[:-2]
    delta_x[-1] = delta_x[-2]
    delta_x[0] = delta_x[1]
    t = np.where(np.abs(delta_x) > np.pi)
    delta_x[t] = np.sign(delta_x[t]) * 2 * np.pi
    delta_x *= 0.5
    return delta_x

def extract_temporal_features(handwritings,features,num=2,finger_mode=False):
    for handwriting in handwritings:
        p = handwriting[:,num]
        x = handwriting[:,0]
        y = handwriting[:,1]
        delta_x = difference(x)
        delta_y = difference(y)
        line_v = line_velocity(x,y)
        line_acc = line_acceleration(x,y)
        angle_v = np.abs(angular_velocity(x,y))
        angle_acc = angular_acceleration(x,y)
        theta = np.arctan2(delta_y,delta_x)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        log_curve_radius = np.log((line_v + 0.05) / (angle_v + 0.05)) # log的曲线弧度
        p_v = difference(p)
        p_acc = difference(p_v)
        delta_v2 = np.abs(line_v * angle_v)
        acceleration = np.sqrt(line_acc ** 2 + delta_v2 ** 2)

        single_feature = np.concatenate((delta_x[:,None],delta_y[:,None],line_v[:,None],line_acc[:,None],
            theta[:,None],cos_theta[:,None],sin_theta[:,None],angle_v[:,None],angle_acc[:,None],
            log_curve_radius[:,None],delta_v2[:,None],acceleration[:,None],p[:,None],p_v[:,None],p_acc[:,None]),axis=1).astype(np.float32)
        if not finger_mode:
            single_feature = (single_feature - np.mean(single_feature,axis=0)) / np.std(single_feature,axis=0)
        else:
            single_feature[:,:-3] = (single_feature[:,:-3] - np.mean(single_feature[:,:-3],axis=0)) / \
                np.std(single_feature[:,:-3],axis=0)
        features.append(single_feature)

def interpolate_torch(org_info,interp_ratio):
    l = len(org_info)
    org_info = torch.tensor(org_info).view(1,1,l,-1)
    new_info = F.interpolate(org_info,size=(l * interp_ratio,3),mode='bicubic').squeeze().numpy()
    return new_info

def save_ckpt(epoch,model,optimizer,lr_scheduler,logger,save_folder,subname,postfix=''):
    save_state = {'model':model.state_dict(),
                  'optimizer':optimizer.state_dict(),
                  'lr_scheduler':lr_scheduler.state_dict(),
                  'epoch':epoch}
    save_path = f'{save_folder}/ckpt-{epoch}-{subname}{postfix}.pth'
    logger.info(f'{save_path} saving checkpoint......')
    torch.save(save_state,save_path)
    logger.info(f'{save_path} successfully saved.')

def load_ckpt(model,pretrained_root,device,logger,optimizer=None,scheduler=None,mode='train',resume=False): 
    state_dict = torch.load(pretrained_root,map_location=device)
    if mode == 'train':
        if resume:
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['lr_scheduler'])
            print(model.load_state_dict(state_dict['model']))
            start_epoch = state_dict['epoch'] + 1
            logger.info(f'mode: "{mode} + resume" {pretrained_root} successfully loaded.')
        else:
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys() and v.numel() == model.state_dict()[k].numel()}
            print(model.load_state_dict(state_dict,strict=False))
            logger.info(f'mode: "{mode} + pretrained" {pretrained_root} successfully loaded.')
            start_epoch = 0
        return start_epoch
    else:
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        # state_dict = {k:v for k,v in state_dict.items() if k in model.state_dict().keys() and v.numel() == model.state_dict()[k].numel()}
        print(model.load_state_dict(state_dict,strict=False))
        # model.load_state_dict(state_dict['model'])
        logger.info(f'mode: "{mode}" {pretrained_root} successfully loaded.')

@lru_cache()
def create_logger(log_root,name='',test=False):
    os.makedirs(f'{log_root}',exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    color_fmt = colored('[%(asctime)s %(name)s]','green') + \
        colored('(%(filename)s %(lineno)d)','yellow') + ': %(levelname)s %(message)s'
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO) # 分布式的等级
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    date = time.strftime('%Y-%m-%d') if not test else time.strftime('%Y-%m-%d') + '-test'
    file_handler = logging.FileHandler(f'{log_root}/log-{date}.txt',mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt,datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger

class LossRecorder:
    def __init__(self,epoch,max_epoch,log_freq,logger,*args):
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.log_freq = log_freq
        self.logger = logger

    def register(self,*args):
        if isinstance(args,tuple):
            for loss in args:
                setattr(self,loss,0.)
        else:
            setattr(self,loss,0.)

    def __repr__(self):
        out = ''
        all_loss = [i for i in dir(self) if 'loss' in i]
        for loss in all_loss:
            name = loss.split('_')
            out += f'{name[-2]}_{name[-1]} '
        return out

    def start(self):
        self.time_start = time.time()

    def update(self,new_loss:dict):
        for k in new_loss.keys():
            l = getattr(self,k)
            setattr(self,k,l + new_loss[k])

    def update_epoch(self,new_epoch):
        self.epoch = new_epoch
        self.reset()

    def info(self):
        all_loss = [i for i in dir(self) if 'loss' in i]
        out = f'epoch: {self.epoch}/{self.max_epoch} '
        for loss in all_loss:
            name = loss.split('_')
            out += f'{name[-2]} {name[-1]}: {getattr(self,loss) / self.log_freq:.4f} '
        out += f'time elasped: {time.time() - self.time_start:.3f}s '
        memory_used = int(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        out += f'memory: {memory_used}MB'
        self.logger.info(out)
    
    def reset(self):
        all_loss = [i for i in dir(self) if 'loss' in i]
        for loss in all_loss:
            setattr(self,loss,0.)
        self.time_start = time.time()

def count_parameters(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param