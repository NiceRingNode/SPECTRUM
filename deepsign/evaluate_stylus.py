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
parser.add_argument('--epoch',type=str,default='30')
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
logger.info(f'epoch: {opt.epoch}')

model = SPECTRUM(15,64,128)
if torch.cuda.is_available():
    torch.cuda.set_device(int(opt.gpu))
    gpu = torch.device(f'cuda:{opt.gpu}')
else:
    gpu = torch.device('cpu')
model = model.to(gpu)

load_ckpt(model,f'../{opt.weights}/ckpt-{opt.epoch}-{settings["name"]}.pth',gpu,logger,mode='test')
model.eval()

sigDict = pickle.load(open("../data/deepsigndb/MCYT_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 25; num_f = 25

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=False)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

logger.info('Computing MCYT')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)
    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs(f'log/seed{opt.seed}/mcyt',exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/mcyt/dtw_dist_p%s.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/mcyt/dtw_dist_n%s.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/mcyt/dtw_dist_temp%s.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/mcyt/dtw_dist_fp{opt.epoch}.npy", dfp)
np.save(f"log/seed{opt.seed}/mcyt/dtw_dist_fn{opt.epoch}.npy", dff)
np.save(f"log/seed{opt.seed}/mcyt/dtw_dist_ftemp{opt.epoch}.npy", dft)


sigDict = pickle.load(open("../data/deepsigndb/BSID_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 16; num_f = 12

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=False)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

logger.info('Computing BiosecurID')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)

    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs(f"log/seed{opt.seed}/bio",exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/bio/dtw_dist_p%s.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/bio/dtw_dist_n%s.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/bio/dtw_dist_temp%s.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/bio/dtw_dist_fp{opt.epoch}.npy", dfp)
np.save(f"log/seed{opt.seed}/bio/dtw_dist_fn{opt.epoch}.npy", dff)
np.save(f"log/seed{opt.seed}/bio/dtw_dist_ftemp{opt.epoch}.npy", dft)


sigDict = pickle.load(open("../data/deepsigndb/BSDS2_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 19; num_f = 20

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=False)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

logger.info('Computing Biosecure DS2')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For BSDB2, 4 templates in session 1 + session2
    idxs = np.concatenate([np.array([0,1,2,3]), np.arange(15, 50)])
    sig = sig[idxs]
    lens = lens[idxs]

    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)

    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs(f'log/seed{opt.seed}/bsds2',exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/bsds2/dtw_dist_p%s.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/bsds2/dtw_dist_n%s.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/bsds2/dtw_dist_temp%s.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/bsds2/dtw_dist_fp{opt.epoch}.npy", dfp)
np.save(f"log/seed{opt.seed}/bsds2/dtw_dist_fn{opt.epoch}.npy", dff)
np.save(f"log/seed{opt.seed}/bsds2/dtw_dist_ftemp{opt.epoch}.npy", dft)


sigDict = pickle.load(open("../data/deepsigndb/EBio2_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 8; num_f = 6

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=False)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

logger.info('Computing eBS DS2 w2')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio2
    sig = torch.from_numpy(sig).to(gpu)
    lens = torch.tensor(lens).long()
    mask = get_len_mask(lens).to(gpu)

    output,_,_,output_freq = model(sig,mask)
    output = output.data.cpu().numpy()
    feats.append(output)
    feat_freq.append(output_freq.detach().cpu().numpy())

os.makedirs(f"log/seed{opt.seed}/ebio2", exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/ebio2/dtw_dist_p%s.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio2/dtw_dist_n%s.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio2/dtw_dist_temp%s.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio2/dtw_dist_fp{opt.epoch}.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio2/dtw_dist_fn{opt.epoch}.npy", dff)
np.save(f"log/seed{opt.seed}/ebio2/dtw_dist_ftemp{opt.epoch}.npy", dft)


sigDict = pickle.load(open("../data/deepsigndb/EBio1_eva.pkl", "rb"), encoding='iso-8859-1')
num_g = 8; num_f = 6

test_dataset = DeepSignDB(sigDict,train=False,finger_mode=False)
test_sampler = TestSamplerDeepSign(test_dataset.config)
dataLoader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

logger.info('Computing eBS DS1 w1')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1
    device = 0
    idxs = np.concatenate([np.array([0,1,10,11,20,21,30,31]) + device * 2,np.array([40,41,42,55,56,57]) + device * 3])
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

os.makedirs(f'log/seed{opt.seed}/ebio1',exist_ok=True)

if opt.rf:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq_rf(feats, feat_freq, opt.template_num, 0, num_g, num_f)
else:
    DIST_P, DIST_N, DIST_TEMP, dfp, dff, dft = dist_seq(feats, feat_freq, opt.template_num, 0, num_g, num_f)
np.save("log/seed%d/ebio1/dtw_dist_p%s_d0.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1/dtw_dist_n%s_d0.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1/dtw_dist_temp%s_d0.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fp{opt.epoch}_d0.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fn{opt.epoch}_d0.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_ftemp{opt.epoch}_d0.npy", dft)


logger.info('Computing eBS DS1 w2')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1
    device = 1
    idxs = np.concatenate([np.array([0,1,10,11,20,21,30,31]) + device * 2,np.array([40,41,42,55,56,57]) + device * 3])
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
np.save("log/seed%d/ebio1/dtw_dist_p%s_d1.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1/dtw_dist_n%s_d1.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1/dtw_dist_temp%s_d1.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fp{opt.epoch}_d1.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fn{opt.epoch}_d1.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_ftemp{opt.epoch}_d1.npy", dft)


logger.info('Computing eBS DS1 w3')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1
    device = 2
    idxs = np.concatenate([np.array([0,1,10,11,20,21,30,31]) + device * 2,np.array([40,41,42,55,56,57]) + device * 3])
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
np.save("log/seed%d/ebio1/dtw_dist_p%s_d2.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1/dtw_dist_n%s_d2.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1/dtw_dist_temp%s_d2.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fp{opt.epoch}_d2.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fn{opt.epoch}_d2.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_ftemp{opt.epoch}_d2.npy", dft)

logger.info('Computing eBS DS1 w4')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1
    device = 3
    idxs = np.concatenate([np.array([0,1,10,11,20,21,30,31]) + device * 2,np.array([40,41,42,55,56,57]) + device * 3])
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
np.save("log/seed%d/ebio1/dtw_dist_p%s_d3.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1/dtw_dist_n%s_d3.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1/dtw_dist_temp%s_d3.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fp{opt.epoch}_d3.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fn{opt.epoch}_d3.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_ftemp{opt.epoch}_d3.npy", dft)


logger.info('Computing eBS DS1 w5')
feats = []
feat_freq = []
for idx, batch in enumerate(dataLoader):
    sig,lens,_,_ = batch

    # For EBio1
    device = 4
    idxs = np.concatenate([np.array([0,1,10,11,20,21,30,31]) + device * 2,np.array([40,41,42,55,56,57]) + device * 3])
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
np.save("log/seed%d/ebio1/dtw_dist_p%s_d4.npy"%(opt.seed, opt.epoch), DIST_P)
np.save("log/seed%d/ebio1/dtw_dist_n%s_d4.npy"%(opt.seed, opt.epoch), DIST_N)
np.save("log/seed%d/ebio1/dtw_dist_temp%s_d4.npy"%(opt.seed, opt.epoch), DIST_TEMP)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fp{opt.epoch}_d4.npy", dfp)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_fn{opt.epoch}_d4.npy", dff)
np.save(f"log/seed{opt.seed}/ebio1/dtw_dist_ftemp{opt.epoch}_d4.npy", dft)