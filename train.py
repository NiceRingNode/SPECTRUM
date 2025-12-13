# -*- coding:utf-8 -*-

import torch
import torch.optim as optimizers
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from model import SPECTRUM,get_len_mask
from dataset import Numbers,TrainSampler,collate_fn,DeepSignDB
from loss import vanilla_triplet_loss
import argparse,os,time,pickle,json
from utils import create_logger,save_ckpt,LossRecorder

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--user_sample',type=int,default=4)
parser.add_argument('--epochs',type=int,default=40)
parser.add_argument('--lr',type=float,default=5e-4)
parser.add_argument('--weight_decay',type=float,default=1e-2)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--cuda',type=bool,default=True)
parser.add_argument('--log_interval',type=int,default=40)
parser.add_argument('--genuine_sample',type=int,default=5)
parser.add_argument('--forgery_sample',type=int,default=10)
parser.add_argument('--folder',type=str,default='data')
parser.add_argument('--round_num',type=str,default='all')
parser.add_argument('--data_name',type=str,default='real')
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--weights_root',type=str,default='./weights')
parser.add_argument('--log_root',type=str,default='./logs')
parser.add_argument('--d_hidden',type=int,default=128)
parser.add_argument('--name',type=str,default='main')
parser.add_argument('--notes',type=str,default='')
opt = parser.parse_args()

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

date_str = f"{time.strftime('%Y%m%d-%H%M%S')}-{opt.name}"
opt.weights_root = f'{opt.weights_root}/{date_str}'
opt.log_root = f"{opt.log_root}/{time.strftime('%Y-%m-%d')}/{date_str}"

os.makedirs(opt.weights_root,exist_ok=True)
os.makedirs(opt.log_root,exist_ok=True)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

logger = create_logger(opt.log_root,name=opt.name)

transform = None

if opt.data_name == 'real' or opt.data_name == 'signature':
    with open(f'./{opt.folder}/{opt.data_name}/{opt.round_num}-{opt.data_name}-train-202.pkl','rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    train_dataset = Numbers(data,transform,train=True)
elif opt.data_name == 'deepsigndb':
    with open(f'./{opt.folder}/{opt.data_name}/MCYT_dev.pkl','rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    train_dataset = DeepSignDB(data,transform,train=True)
    del data
    with open(f'./{opt.folder}/{opt.data_name}/BSID_dev.pkl','rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    train_dataset.add_data(data)
    del data
    with open(f'./{opt.folder}/{opt.data_name}/EBio1_dev.pkl','rb') as f:
        data = pickle.load(f,encoding='iso-8859-1')
    train_dataset.add_data(data)
    del data
else:
    raise NotImplementedError('Unknown data.')

dataset_config = train_dataset.config
train_sampler = TrainSampler(dataset_config,opt.user_sample,opt.genuine_sample,opt.forgery_sample)
train_loader = DataLoader(train_dataset,batch_sampler=train_sampler,collate_fn=collate_fn)

model = SPECTRUM(train_dataset.feature_dims,64,opt.d_hidden)

optimizer = optimizers.AdamW(filter(lambda p:p.requires_grad,model.parameters()),
    lr=opt.lr,weight_decay=opt.weight_decay,eps=1e-8,betas=(0.9,0.999))
scheduler = optimizers.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.epochs,eta_min=opt.lr * 0.001,last_epoch=-1)

if opt.cuda and torch.cuda.is_available():
    torch.cuda.set_device(int(opt.gpu))
    device = torch.device(f'cuda:{opt.gpu}')
else:
    device = torch.device('cpu')
model = model.to(device)

prob_loss = torch.nn.BCELoss().to(device)
start_epoch = 0
loss_recorder = LossRecorder(start_epoch,opt.epochs,opt.log_interval,logger)
loss_recorder.register(
    'epoch_intra_loss',
    'epoch_outra_loss',
    'epoch_bin_loss'
)

logger.info(opt)
logger.info(f'data root: ./{opt.folder}/{opt.data_name}/{opt.round_num}-{opt.data_name}\n'
    f'input dimensions: {train_dataset.feature_dims}\n'
    f'train loader length: {len(train_loader)} train features length: {len(train_dataset)}\n'
    f'model: {model.__class__.__name__}\noptimizer: {optimizer.__class__.__name__}\n'
    f'scheduler: {scheduler.__class__.__name__}\nloss: {loss_recorder}\nnotes: {opt.notes}')
with open(f'{opt.weights_root}/settings.json','w',encoding='utf-8') as f:
    f.write(json.dumps({**vars(opt)},indent=4,ensure_ascii=False))

def train():
    try:
        model.train()
        loss_recorder.start()
        for epoch in range(start_epoch,opt.epochs):
            loss_recorder.update_epoch(epoch)
            for i,(features,features_lens,_,binary_labels) in enumerate(train_loader):
                features = torch.from_numpy(features).to(device)
                features_lens = torch.tensor(features_lens).long().to(device)
                binary_labels = torch.tensor(binary_labels).float().to(device)
                mask = get_len_mask(features_lens).to(device)
                y_vector,out_features_lens,freq_logits,_ = model(features,mask)
                cur_intra_loss,cur_outra_loss = vanilla_triplet_loss(y_vector,
                    opt.genuine_sample,opt.forgery_sample,out_features_lens)
                cur_bin_loss = prob_loss(freq_logits,binary_labels.view(-1,1))
                optimizer.zero_grad()
                (cur_intra_loss * 0.01 + cur_outra_loss + cur_bin_loss).backward()
                optimizer.step()
                loss_recorder.update({
                    'epoch_intra_loss':cur_intra_loss.item(),
                    'epoch_outra_loss':cur_outra_loss.item(),
                    'epoch_bin_loss':cur_bin_loss.item()
                })
                if i % opt.log_interval == 0 and i != 0:
                    loss_recorder.info()
                    loss_recorder.reset()
            save_ckpt(epoch,model,optimizer,scheduler,logger,opt.weights_root,opt.name)
            scheduler.step()
    except KeyboardInterrupt:
        save_ckpt(opt.epochs + 1,model,optimizer,scheduler,logger,opt.weights_root,opt.name)

def main():
    train()

if __name__ == '__main__':
    main()