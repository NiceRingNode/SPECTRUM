# -*- coding:utf-8 -*-

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from model import SPECTRUM,get_len_mask
from dataset import Numbers,TestSampler,collate_fn
import numpy as np
from utils import create_logger,load_ckpt,count_parameters
from dist import dist_skilled_forgery,dist_random_forgery
from verify import verify
import argparse,os,time,pickle,json

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--train_user_number',type=int,default=202)
parser.add_argument('--seed',type=int,default=123)
parser.add_argument('--cuda',type=bool,default=True)
parser.add_argument('--genuine_sample',type=int,default=20)
parser.add_argument('--forgery_sample',type=int,default=20)
parser.add_argument('--folder',type=str,default='data')
parser.add_argument('--round_num',type=str,default='all')
parser.add_argument('--data_name',type=str,default='real')
parser.add_argument('--weights_root',type=str,default='')
parser.add_argument('--output_root',type=str,default='./output')
parser.add_argument('--log_root',type=str,default='./logs')
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--epoch',type=int,default=39)
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--template_num',type=int,default=4)
parser.add_argument('--d_hidden',type=int,default=128)
parser.add_argument('--test_all_eer',action='store_true')
parser.add_argument('--name',type=str,default='')
parser.add_argument('--notes',type=str,default='')
opt = parser.parse_args()

with open(f'{opt.weights_root}/settings.json','r',encoding='utf-8') as f:
    settings = json.loads(f.read())
opt.seed = settings['seed']
opt.name = settings['name']
opt.notes = settings['notes']
opt.d_hidden = settings['d_hidden']
opt.data_name = settings['data_name']
opt.round_num = settings['round_num']
opt.log_root = settings['log_root']
if opt.round_num == 'all':
    opt.genuine_sample = 20
    opt.forgery_sample = 20
else:
    opt.genuine_sample = 10
    opt.forgery_sample = 10

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

os.makedirs(opt.log_root,exist_ok=True)
os.makedirs(opt.output_root,exist_ok=True)

sample_interval = opt.genuine_sample + opt.forgery_sample
logger = create_logger(opt.log_root,name=opt.name,test=True)
with open(f'./{opt.folder}/{opt.data_name}/{opt.round_num}-{opt.data_name}-test-200.pkl','rb') as f:
    handwriting_info = pickle.load(f,encoding='iso-8859-1')
test_dataset = Numbers(handwriting_info,train=False)
users_cnt = test_dataset.config['users_cnt']
test_sampler = TestSampler(users_cnt,opt.genuine_sample,opt.forgery_sample)
test_loader = DataLoader(test_dataset,batch_sampler=test_sampler,collate_fn=collate_fn)

model = SPECTRUM(test_dataset.feature_dims,64,128)


logger.info(f'./{opt.folder}/{opt.data_name}/{opt.round_num}-{opt.data_name}-test-200.pkl\n'
    f'test loader length: {len(test_loader)}\n'
    f'genuine_sample: {opt.genuine_sample}\nforgery_sample: {opt.forgery_sample}\n'
    f'sample_interval: {sample_interval}\nseed: {opt.seed}\n'
    f'model: {model.__class__.__name__}\nnotes: {opt.notes}')

if opt.cuda and torch.cuda.is_available():
    torch.cuda.set_device(int(opt.gpu))
    device = torch.device(f'cuda:{opt.gpu}')
else:
    device = torch.device('cpu')
model = model.to(device)

@torch.no_grad()
def test():
    logger.info('<======test begins.======>')
    model.eval()
    output,output_freq = [],[]
    start = time.time()
    for i,(features,features_lens,_,_) in enumerate(test_loader):
        features = torch.from_numpy(features).to(device)
        features_lens = torch.tensor(features_lens).long().to(device)
        mask = get_len_mask(features_lens).to(device)
        y_vector,_,_,y_freq = model(features,mask)
        output.append(y_vector.cpu().numpy())
        output_freq.append(y_freq.cpu().numpy())
    end = time.time()
    extraction_period = end - start
    logger.info(f'feature extraction time per sample: {1000 * extraction_period / len(test_dataset):.3f} ms/s')

    skilled_start = time.time()
    dist_genuine,dist_forgery,dist_template,dfg,dff,dft = dist_skilled_forgery(output,output_freq,opt.template_num,0,opt.genuine_sample,opt.forgery_sample)
    np.save(f'{opt.output_root}/dist_genuine.npy',dist_genuine)
    np.save(f'{opt.output_root}/dist_forgery.npy',dist_forgery)
    np.save(f'{opt.output_root}/dist_template.npy',dist_template)
    np.save(f'{opt.output_root}/dist_fgenuine.npy',dfg)
    np.save(f'{opt.output_root}/dist_fforgery.npy',dff)
    np.save(f'{opt.output_root}/dist_ftemplate.npy',dft)
    verify(logger,opt.template_num,users_cnt,opt.genuine_sample,opt.forgery_sample,rf=False)
    skilled_end = time.time()
    skilled_period = skilled_end - skilled_start
    logger.info(f'skilled forgery verification time per sample: {1000 * skilled_period / len(test_dataset):.3f} ms/s')
    logger.info(f'skilled forgery verification time per sample (total): {1000 * (skilled_period + extraction_period) / len(test_dataset):.3f} ms/s')

    random_start = time.time()
    dist_genuine,dist_forgery,dist_template,dfg,dff,dft = dist_random_forgery(output,output_freq,opt.template_num,0,opt.genuine_sample)
    np.save(f'{opt.output_root}/dist_genuine.npy',dist_genuine)
    np.save(f'{opt.output_root}/dist_forgery.npy',dist_forgery)
    np.save(f'{opt.output_root}/dist_template.npy',dist_template)
    np.save(f'{opt.output_root}/dist_fgenuine.npy',dfg)
    np.save(f'{opt.output_root}/dist_fforgery.npy',dff)
    np.save(f'{opt.output_root}/dist_ftemplate.npy',dft)
    verify(logger,opt.template_num,users_cnt,opt.genuine_sample,opt.forgery_sample,rf=True)
    random_end = time.time()
    random_period = random_end - random_start
    logger.info(f'random forgery verification time per sample: {1000 * random_period / len(test_dataset):.3f} ms/s')
    logger.info(f'random forgery verification time per sample (total): {1000 * (random_period + extraction_period) / len(test_dataset):.3f} ms/s')

    _,params = count_parameters(model)
    logger.info(f'params: {params / 1e6:.6f}')

@torch.no_grad()
def test_all(template_num):
    logger.info('<======test all begins.======>')
    model.eval()
    output,output_freq = [],[]
    for i,(features,features_lens,_,_) in enumerate(test_loader):
        features = torch.from_numpy(features).to(device)
        features_lens = torch.tensor(features_lens).long().to(device)
        mask = get_len_mask(features_lens).to(device)
        y_vector,_,_,y_freq = model(features,mask)
        output.append(y_vector.cpu().numpy())
        output_freq.append(y_freq.cpu().numpy())

    if template_num == 4 or template_num == 1:
        dist_genuine,dist_forgery,dist_template,dfg,dff,dft = dist_skilled_forgery(output,output_freq,template_num,
            0,opt.genuine_sample,opt.forgery_sample)
        np.save(f'{opt.output_root}/dist_genuine.npy',dist_genuine)
        np.save(f'{opt.output_root}/dist_forgery.npy',dist_forgery)
        np.save(f'{opt.output_root}/dist_template.npy',dist_template)
        np.save(f'{opt.output_root}/dist_fgenuine.npy',dfg)
        np.save(f'{opt.output_root}/dist_fforgery.npy',dff)
        np.save(f'{opt.output_root}/dist_ftemplate.npy',dft)
        verify(logger,template_num,users_cnt,opt.genuine_sample,opt.forgery_sample,rf=False)

        dist_genuine,dist_forgery,dist_template,dfg,dff,dft = dist_random_forgery(output,output_freq,template_num,
            0,opt.genuine_sample)
        np.save(f'{opt.output_root}/dist_genuine.npy',dist_genuine)
        np.save(f'{opt.output_root}/dist_forgery.npy',dist_forgery)
        np.save(f'{opt.output_root}/dist_template.npy',dist_template)
        np.save(f'{opt.output_root}/dist_fgenuine.npy',dfg)
        np.save(f'{opt.output_root}/dist_fforgery.npy',dff)
        np.save(f'{opt.output_root}/dist_ftemplate.npy',dft)
        verify(logger,template_num,users_cnt,opt.genuine_sample,opt.forgery_sample,rf=True)
    else:
        dist_genuine,dist_forgery,dist_template,dfg,dff,dft = dist_skilled_forgery(output,output_freq,template_num,
            0,opt.genuine_sample,opt.forgery_sample)
        np.save(f'{opt.output_root}/dist_genuine.npy',dist_genuine)
        np.save(f'{opt.output_root}/dist_forgery.npy',dist_forgery)
        np.save(f'{opt.output_root}/dist_template.npy',dist_template)
        np.save(f'{opt.output_root}/dist_fgenuine.npy',dfg)
        np.save(f'{opt.output_root}/dist_fforgery.npy',dff)
        np.save(f'{opt.output_root}/dist_ftemplate.npy',dft)
        verify(logger,template_num,users_cnt,opt.genuine_sample,opt.forgery_sample,rf=False)

def main():
    if not opt.test_all_eer:
        load_ckpt(model,f'{opt.weights_root}/ckpt-{opt.epoch}-{opt.name}.pth',device,logger,mode='test')
        time_elapsed_start = time.time()
        test()
        logger.info(f'time elapsed: {time.time() - time_elapsed_start:.5f}s\n')
    else:
        load_ckpt(model,f'{opt.weights_root}/ckpt-{opt.epoch}-{opt.name}.pth',device,logger,mode='test')
        for i in range(4,0,-1):
            time_elapsed_start = time.time()
            test_all(i)
            logger.info(f'time elapsed: {time.time() - time_elapsed_start:.5f}s\n')

if __name__ == '__main__':
    main()