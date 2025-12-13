# -*- coding:utf-8 -*-

import os,pickle,argparse,json,sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dist import dist_seq,dist_seq_rf
sys.path.append('..')
from model import SPECTRUM,get_len_mask
from utils import load_ckpt,create_logger
from dataset import DeepSignDB,TestSamplerDeepSign,collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--template_num',type=int,default=4)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--epoch',type=str,default="End")
parser.add_argument('--weights',type=str,default='')
parser.add_argument('--gpu',type=str,default='0')
parser.add_argument('--rf',action='store_true')
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

with open(f'../{opt.weights}/settings.json','r',encoding='utf-8') as f:
    settings = json.loads(f.read())
logger = create_logger(settings['log_root'],name=settings['name'],test=True)

logger.info('skilled forgery' if not opt.rf else 'random forgery')

model = SPECTRUM(15,64,128)
if torch.cuda.is_available():
    torch.cuda.set_device(int(opt.gpu))
    gpu = torch.device(f'cuda:{opt.gpu}')
else:
    gpu = torch.device('cpu')
model = model.to(gpu)

load_ckpt(model,f'../{opt.weights}/ckpt-{opt.epoch}-{settings["name"]}.pth',gpu,logger,mode='test')
model.eval()

sigDict = pickle.load(open("../data/deepsigndb/EBio1_eva_finger.pkl", "rb"),encoding="iso-8859-1")
num_g = 8; num_f = 6

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=True)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

logger.info('Computing eBS DS1 w4, finger inputs')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1, finger
    device = 0
    idxs = np.concatenate([np.array([0,1,4,5,8,9,12,13]) + device * 2,np.array([16,17,18,22,23,24]) + device * 3])
    sig = sig[idxs]
    lens = lens[idxs]
    sig = sig[:,0:int(np.max(lens)),:]
    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)
    
    output,_,_,output_freq = model(sig,mask)
    # output,_,_ = model(sig,lens)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs("log/seed%d"%opt.seed,exist_ok=True)
os.makedirs("log/seed%d/ebio1_finger"%opt.seed,exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/ebio1_finger/dtw_dist_p%s_d0.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1_finger/dtw_dist_n%s_d0.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1_finger/dtw_dist_temp%s_d0.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1_finger/dtw_dist_fp{opt.epoch}_d0.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1_finger/dtw_dist_fn{opt.epoch}_d0.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1_finger/dtw_dist_ftemp{opt.epoch}_d0.npy", dft)

logger.info('Computing eBS DS1 w5, finger inputs')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1, finger
    device = 1
    idxs = np.concatenate([np.array([0,1,4,5,8,9,12,13]) + device * 2,
                           np.array([16,17,18,22,23,24]) + device * 3])
    sig = sig[idxs]
    lens = lens[idxs]
    sig = sig[:,0:int(np.max(lens)),:]
    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)

    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs(f'log/seed{opt.seed}/ebio1_finger',exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/ebio1_finger/dtw_dist_p%s_d1.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1_finger/dtw_dist_n%s_d1.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1_finger/dtw_dist_temp%s_d1.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1_finger/dtw_dist_fp{opt.epoch}_d1.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1_finger/dtw_dist_fn{opt.epoch}_d1.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1_finger/dtw_dist_ftemp{opt.epoch}_d1.npy", dft)


sigDict = pickle.load(open("../data/deepsigndb/EBio2_eva_finger.pkl", "rb"), encoding="iso-8859-1")
num_g = 8; num_f = 6

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=True)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

logger.info('Computing eBS DS2 w5, finger inputs')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio2, finger
    device = 0
    if device == 0:
        idxs = np.concatenate([np.array([0,1,4,5,8,9,10,11]), np.array([16,17,18,22,23,24])])
    else:
        idxs = np.concatenate([np.array([2,3,6,7,12,13,14,15]), np.array([19,20,21,25,26,27])])
    sig = sig[idxs]
    lens = lens[idxs]
    sig = sig[:,0:int(np.max(lens)),:]
    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)

    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs(f'log/seed{opt.seed}/ebio2_finger',exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/ebio2_finger/dtw_dist_p%s_d0.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio2_finger/dtw_dist_n%s_d0.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio2_finger/dtw_dist_temp%s_d0.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio2_finger/dtw_dist_fp{opt.epoch}_d0.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio2_finger/dtw_dist_fn{opt.epoch}_d0.npy", dff)
np.save(f"log/seed{opt.seed}/ebio2_finger/dtw_dist_ftemp{opt.epoch}_d0.npy", dft)

logger.info('Computing eBS DS2 w6, finger inputs')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio2, finger
    device = 1
    if device == 0:
        idxs = np.concatenate([np.array([0,1,4,5,8,9,10,11]), np.array([16,17,18,22,23,24])])
    else:
        idxs = np.concatenate([np.array([2,3,6,7,12,13,14,15]), np.array([19,20,21,25,26,27])])

    sig = sig[idxs]
    lens = lens[idxs]
    sig = sig[:,0:int(np.max(lens)),:]
    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)

    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/ebio2_finger/dtw_dist_p%s_d1.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio2_finger/dtw_dist_n%s_d1.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio2_finger/dtw_dist_temp%s_d1.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio2_finger/dtw_dist_fp{opt.epoch}_d1.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio2_finger/dtw_dist_fn{opt.epoch}_d1.npy", dff)
np.save(f"log/seed{opt.seed}/ebio2_finger/dtw_dist_ftemp{opt.epoch}_d1.npy", dft)