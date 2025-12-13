# -*- coding:utf-8 -*-

import json,sys,argparse
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('..')
from utils import create_logger

parser = argparse.ArgumentParser()
parser.add_argument('--template_num',type=int,default=4)
parser.add_argument('--epoch',type=str,default='39')
parser.add_argument('--weights',type=str,default='')
opt = parser.parse_args()
template_num = opt.template_num
epoch = opt.epoch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

with open(f'../{opt.weights}/settings.json','r',encoding='utf-8') as f:
    settings = json.loads(f.read())
logger = create_logger(settings['log_root'],name=settings['name'],test=True)

def select_template(dist_matrix):
    template_num = len(dist_matrix)
    if template_num == 1:
        return None,1,1,1,1
    dist_matrix = (dist_matrix + dist_matrix.transpose())
    dist_avg = np.sum(dist_matrix,axis=1) / (template_num - 1)
    min_idx = np.argmin(dist_avg)
    dist_var = np.sqrt((np.sum(np.power(dist_matrix,2)) / template_num / (template_num - 1) - \
        np.power(np.sum(dist_matrix) / template_num / (template_num - 1),2)))
    dist_mean = np.sum(dist_matrix) / template_num / (template_num - 1)
    dist_temp = np.sum(dist_matrix[:,min_idx]) / (template_num - 1)
    dist_max = np.mean(np.max(dist_matrix,axis=1))
    dist_matrix[range(template_num),range(template_num)] = np.inf
    dist_matrix[dist_matrix == 0] = np.inf
    dist_min = np.mean(np.min(dist_matrix,axis=1))
    return min_idx,dist_temp ** 0.5,dist_max ** 0.5,dist_min ** 0.5,dist_mean ** 0.5

def getEER(FAR, FRR):
    a = FRR <= FAR
    s = np.sum(a)
    a[-s-1] = 1
    a[-s+1:] = 0
    FRR = FRR[a]
    FAR = FAR[a] 
    a = [[FRR[1]-FRR[0], FAR[0]-FAR[1]], [-1, 1]]
    b = [(FRR[1]-FRR[0])*FAR[0]-(FAR[1]-FAR[0])*FRR[0], 0]
    return np.linalg.solve(a, b)

def scoreScatter(gen, forg):
    ax = plt.subplot()
    ax.scatter(gen[:,0],gen[:,1], color='r')
    ax.scatter(forg[:,0],forg[:,1], color='k', marker="*")
    ax.set_xlabel("score$_{min}$", fontsize=20)
    ax.set_ylabel("score$_{ave}$", fontsize=20)
    k = (np.sum(gen[:,1] / gen[:,0]) + np.sum(forg[:,1] / forg[:,0])) / (forg.shape[0] + gen.shape[0])
    x = np.linspace(0, 0.5, 500)  
    y = -x / k + 0.5
    plt.plot(x, y)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("RAN$_r$ verification scores", fontsize=20)
    plt.show()

if template_num == 4:
    logger.info('4 templates')
    LOOP = False
elif template_num == 1:
    logger.info('1 template')
    LOOP = True
else:
    raise ValueError

EER_all,mEER_L = [],[]

n_users = 35
n_test_g = 8 - 4
n_test_f = 6
N_TEST_G = n_users * n_test_g
N_TEST_F = n_users * n_test_f

ROC_FAR = 0
ROC_FRR = 0
TOTAL_P = 0
TOTAL_N = 0

EER_G = []; EER_L = []
logger.info("For eBS DS1 w4, finger inputs:")
seed = 123
DIST_P = np.load("log/seed%d/ebio1_finger/dtw_dist_p%s_d0.npy"%(seed, epoch))
DIST_N = np.load("log/seed%d/ebio1_finger/dtw_dist_n%s_d0.npy"%(seed, epoch))
DIST_TEMP = np.load("log/seed%d/ebio1_finger/dtw_dist_temp%s_d0.npy"%(seed, epoch))
dist_fgenuine = np.load("log/seed%d/ebio1_finger/dtw_dist_fp%s_d0.npy"%(seed, epoch))
dist_fforgery = np.load("log/seed%d/ebio1_finger/dtw_dist_fn%s_d0.npy"%(seed, epoch))
dist_ftemplate = np.load("log/seed%d/ebio1_finger/dtw_dist_ftemp%s_d0.npy"%(seed, epoch))
if LOOP:
    for i in range(4):
        datum_p = []
        datum_n = []
        EERs = []
        datum_fp,datum_fn = [],[]
        for ii in range(n_users):   
            dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            
            dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1)

            datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

            dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)

            dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)

            datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
            datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

        datum_p = np.concatenate(datum_p, axis=0)
        datum_n = np.concatenate(datum_n, axis=0)
        datum_fp = np.concatenate(datum_fp, axis=0)
        datum_fn = np.concatenate(datum_fn, axis=0)

        for ii in range(n_users):    
            k = 1 #Simply set to 1.
            c = np.arange(0, 50, 0.002)[None,:]
            FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,1]),1 - sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,1]),1 - sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(np.mean(EERs))

        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 1 * 0.25 * datum_n.shape[0]
        ROC_FRR += FRR * 1 * 0.25 * datum_p.shape[0]
else:
    datum_p = []
    datum_n = []
    EERs = []
    datum_fp,datum_fn = [],[]
    for ii in range(n_users):   
        idx, dtmp, dmax, dmin, dmean = select_template(DIST_TEMP[ii*template_num:(ii+1)*template_num,0:template_num])
        
        dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmax 
        dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmin 
        dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmean 

        dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmax 
        dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmin 
        dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmean

        datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.)
        datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.)

        dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmax / 2
        dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmin / 2
        dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmean / 2

        dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmax / 2
        dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmin / 2
        dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmean / 2

        datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
        datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

    datum_p = np.concatenate(datum_p, axis=0)
    datum_n = np.concatenate(datum_n, axis=0)
    datum_fp = np.concatenate(datum_fp, axis=0)
    datum_fn = np.concatenate(datum_fn, axis=0)

    for ii in range(n_users): 
        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,1]),1 - sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
        FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,1]),1 - sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,2])],axis=-1),axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
        EERs.append(getEER(FAR, FRR)[0] * 100)
    EER_L.append(np.mean(EERs))

    k = 1.
    c = np.arange(0, 50, 0.002)[None,:]
    FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
    FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
    EER_G.append(getEER(FAR, FRR)[0] * 100)

    ROC_FAR += FAR * 1 * datum_n.shape[0]
    ROC_FRR += FRR * 1 * datum_p.shape[0]

logger.info(f'Global threshold: {np.mean(EER_G):.5f}, Local threshold: {np.mean(EER_L):.5f}')
mEER_L.append(np.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

EER_G = []; EER_L = []
logger.info("For eBS DS1 w5, finger inputs:")
DIST_P = np.load("log/seed%d/ebio1_finger/dtw_dist_p%s_d1.npy"%(seed, epoch))
DIST_N = np.load("log/seed%d/ebio1_finger/dtw_dist_n%s_d1.npy"%(seed, epoch))
DIST_TEMP = np.load("log/seed%d/ebio1_finger/dtw_dist_temp%s_d1.npy"%(seed, epoch))
dist_fgenuine = np.load("log/seed%d/ebio1_finger/dtw_dist_fp%s_d1.npy"%(seed, epoch))
dist_fforgery = np.load("log/seed%d/ebio1_finger/dtw_dist_fn%s_d1.npy"%(seed, epoch))
dist_ftemplate = np.load("log/seed%d/ebio1_finger/dtw_dist_ftemp%s_d1.npy"%(seed, epoch))
if LOOP:
    for i in range(4):
        datum_p = []
        datum_n = []
        EERs = []
        datum_fp,datum_fn = [],[]
        for ii in range(n_users):   
            dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            
            dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1)

            datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

            dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)

            dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)

            datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
            datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

        datum_p = np.concatenate(datum_p, axis=0)
        datum_n = np.concatenate(datum_n, axis=0)
        datum_fp = np.concatenate(datum_fp, axis=0)
        datum_fn = np.concatenate(datum_fn, axis=0)

        for ii in range(n_users):    
            k = 1
            c = np.arange(0, 50, 0.002)[None,:]
            FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,1]),1 - sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,1]),1 - sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(np.mean(EERs))

        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 1 * 0.25 * datum_n.shape[0]
        ROC_FRR += FRR * 1 * 0.25 * datum_p.shape[0]
else:
    datum_p = []
    datum_n = []
    EERs = []
    datum_fp,datum_fn = [],[]
    for ii in range(n_users):   
        idx, dtmp, dmax, dmin, dmean = select_template(DIST_TEMP[ii*template_num:(ii+1)*template_num,0:template_num])
        
        dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmax 
        dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmin 
        dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmean 

        dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmax 
        dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmin 
        dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmean

        datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.)
        datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.)

        dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmax / 2
        dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmin / 2
        dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmean / 2

        dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmax / 2
        dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmin / 2
        dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmean / 2

        datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
        datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

    datum_p = np.concatenate(datum_p, axis=0)
    datum_n = np.concatenate(datum_n, axis=0)
    datum_fp = np.concatenate(datum_fp, axis=0)
    datum_fn = np.concatenate(datum_fn, axis=0)

    for ii in range(n_users): 
        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,1]),1 - sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
        FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,1]),1 - sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,2])],axis=-1),axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
        EERs.append(getEER(FAR, FRR)[0] * 100)
    EER_L.append(np.mean(EERs))

    k = 1.
    c = np.arange(0, 50, 0.002)[None,:]
    FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
    FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
    EER_G.append(getEER(FAR, FRR)[0] * 100)

    ROC_FAR += FAR * 1 * datum_n.shape[0]
    ROC_FRR += FRR * 1 * datum_p.shape[0]

logger.info(f'Global threshold: {np.mean(EER_G):.5f}, Local threshold: {np.mean(EER_L):.5f}')
mEER_L.append(np.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

EER_G = []; EER_L = []
logger.info("For eBS DS2 w5, finger inputs:")
DIST_P = np.load("log/seed%d/ebio2_finger/dtw_dist_p%s_d0.npy"%(seed, epoch))
DIST_N = np.load("log/seed%d/ebio2_finger/dtw_dist_n%s_d0.npy"%(seed, epoch))
DIST_TEMP = np.load("log/seed%d/ebio2_finger/dtw_dist_temp%s_d0.npy"%(seed, epoch))
dist_fgenuine = np.load("log/seed%d/ebio2_finger/dtw_dist_fp%s_d0.npy"%(seed, epoch))
dist_fforgery = np.load("log/seed%d/ebio2_finger/dtw_dist_fn%s_d0.npy"%(seed, epoch))
dist_ftemplate = np.load("log/seed%d/ebio2_finger/dtw_dist_ftemp%s_d0.npy"%(seed, epoch))
if LOOP:
    for i in range(4):
        datum_p = []
        datum_n = []
        EERs = []
        datum_fp,datum_fn = [],[]
        for ii in range(n_users):   
            dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            
            dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1)

            datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

            dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)

            dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)

            datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
            datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

        datum_p = np.concatenate(datum_p, axis=0)
        datum_n = np.concatenate(datum_n, axis=0)
        datum_fp = np.concatenate(datum_fp, axis=0)
        datum_fn = np.concatenate(datum_fn, axis=0)

        for ii in range(n_users):    
            k = 1
            c = np.arange(0, 50, 0.002)[None,:]
            FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,1]),1 - sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,1]),1 - sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(np.mean(EERs))

        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 1 * 0.25 * datum_n.shape[0]
        ROC_FRR += FRR * 1 * 0.25 * datum_p.shape[0]
else:
    datum_p = []
    datum_n = []
    EERs = []
    datum_fp,datum_fn = [],[]
    for ii in range(n_users):   
        idx, dtmp, dmax, dmin, dmean = select_template(DIST_TEMP[ii*template_num:(ii+1)*template_num,0:template_num])
        
        dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmax 
        dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmin 
        dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmean 

        dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmax 
        dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmin 
        dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmean

        datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.)
        datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.)

        dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmax / 2
        dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmin / 2
        dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmean / 2

        dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmax / 2
        dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmin / 2
        dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmean / 2

        datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
        datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

    datum_p = np.concatenate(datum_p, axis=0)
    datum_n = np.concatenate(datum_n, axis=0)
    datum_fp = np.concatenate(datum_fp, axis=0)
    datum_fn = np.concatenate(datum_fn, axis=0)

    for ii in range(n_users): 
        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,1]),1 - sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
        FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,1]),1 - sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,2])],axis=-1),axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
        EERs.append(getEER(FAR, FRR)[0] * 100)
    EER_L.append(np.mean(EERs))

    k = 1.
    c = np.arange(0, 50, 0.002)[None,:]
    FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
    FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
    EER_G.append(getEER(FAR, FRR)[0] * 100)

    ROC_FAR += FAR * 1 * datum_n.shape[0]
    ROC_FRR += FRR * 1 * datum_p.shape[0]

logger.info(f'Global threshold: {np.mean(EER_G):.5f}, Local threshold: {np.mean(EER_L):.5f}')
mEER_L.append(np.mean(EER_L))
TOTAL_P += datum_p.shape[0] 
TOTAL_N += datum_n.shape[0] 

EER_G = []; EER_L = []
logger.info("For eBS DS2 w6, finger inputs:")
DIST_P = np.load("log/seed%d/ebio2_finger/dtw_dist_p%s_d1.npy"%(seed, epoch))
DIST_N = np.load("log/seed%d/ebio2_finger/dtw_dist_n%s_d1.npy"%(seed, epoch))
DIST_TEMP = np.load("log/seed%d/ebio2_finger/dtw_dist_temp%s_d1.npy"%(seed, epoch))
dist_fgenuine = np.load("log/seed%d/ebio2_finger/dtw_dist_fp%s_d1.npy"%(seed, epoch))
dist_fforgery = np.load("log/seed%d/ebio2_finger/dtw_dist_fn%s_d1.npy"%(seed, epoch))
dist_ftemplate = np.load("log/seed%d/ebio2_finger/dtw_dist_ftemp%s_d1.npy"%(seed, epoch))
if LOOP:
    for i in range(4):
        datum_p = []
        datum_n = []
        EERs = []
        datum_fp,datum_fn = [],[]
        for ii in range(n_users):   
            dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i+1], axis=1) 
            
            dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1) 
            dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i+1], axis=1)

            datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.) 
            datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.) 

            dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)
            dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,i:i + 1] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,i:i + 1],axis=1)

            dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)
            dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,i:i + 1] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,i:i + 1],axis=1)

            datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
            datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

        datum_p = np.concatenate(datum_p, axis=0)
        datum_n = np.concatenate(datum_n, axis=0)
        datum_fp = np.concatenate(datum_fp, axis=0)
        datum_fn = np.concatenate(datum_fn, axis=0)

        for ii in range(n_users):    
            k = 1
            c = np.arange(0, 50, 0.002)[None,:]
            FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,1]),1 - sigmoid(datum_fp[ii*n_test_g:(ii+1)*n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
            FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * \
                np.stack([1 + sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,1]),1 - sigmoid(datum_fn[ii*n_test_f:(ii+1)*n_test_f,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
            EERs.append(getEER(FAR, FRR)[0] * 100)
        EER_L.append(np.mean(EERs))

        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
        FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
        EER_G.append(getEER(FAR, FRR)[0] * 100)

        ROC_FAR += FAR * 1 * 0.25 * datum_n.shape[0]
        ROC_FRR += FRR * 1 * 0.25 * datum_p.shape[0]
else:
    datum_p = []
    datum_n = []
    EERs = []
    datum_fp,datum_fn = [],[]
    for ii in range(n_users):   
        idx, dtmp, dmax, dmin, dmean = select_template(DIST_TEMP[ii*template_num:(ii+1)*template_num,0:template_num])
        
        dmax_p = np.max(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmax 
        dmin_p = np.min(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmin 
        dmean_p = np.mean(DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num], axis=1) / dmean 

        dmax_n = np.max(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmax 
        dmin_n = np.min(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmin 
        dmean_n = np.mean(DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num], axis=1) / dmean

        datum_p.append(np.concatenate((dmax_p[:,None], dmin_p[:,None], dmean_p[:,None]), axis=1) / 10.)
        datum_n.append(np.concatenate((dmax_n[:,None], dmin_n[:,None], dmean_n[:,None]), axis=1) / 10.)

        dmax_fp = np.max(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmax / 2
        dmin_fp = np.min(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmin / 2
        dmean_fp = np.mean(dist_fgenuine[ii * n_test_g:(ii + 1) * n_test_g,0:template_num] + DIST_P[ii*n_test_g:(ii+1)*n_test_g,0:template_num],axis=1) / dmean / 2

        dmax_fn = np.max(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmax / 2
        dmin_fn = np.min(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmin / 2
        dmean_fn = np.mean(dist_fforgery[ii * n_test_f:(ii + 1) * n_test_f,0:template_num] + DIST_N[ii*n_test_f:(ii+1)*n_test_f,0:template_num],axis=1) / dmean / 2

        datum_fp.append(np.concatenate((dmax_fp[:,None],dmin_fp[:,None],dmean_fp[:,None]),axis=1) / 10.)
        datum_fn.append(np.concatenate((dmax_fn[:,None],dmin_fn[:,None],dmean_fn[:,None]),axis=1) / 10.)

    datum_p = np.concatenate(datum_p, axis=0)
    datum_n = np.concatenate(datum_n, axis=0)
    datum_fp = np.concatenate(datum_fp, axis=0)
    datum_fn = np.concatenate(datum_fn, axis=0)

    for ii in range(n_users): 
        k = 1
        c = np.arange(0, 50, 0.002)[None,:]
        FRR = 1. - np.sum(np.sum(datum_p[ii*n_test_g:(ii+1)*n_test_g,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,1]),1 - sigmoid(datum_fp[ii * n_test_g:(ii + 1) * n_test_g,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(n_test_g)
        FAR = 1. - np.sum(np.sum(datum_n[ii*n_test_f:(ii+1)*n_test_f,1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,1]),1 - sigmoid(datum_fn[ii * n_test_f:(ii + 1) * n_test_f,2])],axis=-1),axis=1)[:,None] - c >= 0, axis=0) / float(n_test_f)
        EERs.append(getEER(FAR, FRR)[0] * 100)
    EER_L.append(np.mean(EERs))

    k = 1.
    c = np.arange(0, 50, 0.002)[None,:]
    FRR = 1. - np.sum(np.sum(datum_p[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fp[:,1]),1 - sigmoid(datum_fp[:,2])],axis=-1), axis=1)[:,None] - c <= 0, axis=0) / float(datum_p.shape[0])
    FAR = 1. - np.sum(np.sum(datum_n[:, 1:] * [1, 1/k] * np.stack([1 + sigmoid(datum_fn[:,1]),1 - sigmoid(datum_fn[:,2])],axis=-1), axis=1)[:,None] - c >= 0, axis=0) / float(datum_n.shape[0])
    EER_G.append(getEER(FAR, FRR)[0] * 100)

    ROC_FAR += FAR * 1 * datum_n.shape[0]
    ROC_FRR += FRR * 1 * datum_p.shape[0]

logger.info(f'Global threshold: {np.mean(EER_G):.5f}, Local threshold: {np.mean(EER_L):.5f}')
mEER_L.append(np.mean(EER_L))
TOTAL_P += datum_p.shape[0]
TOTAL_N += datum_n.shape[0]

final_global = getEER(ROC_FAR*1.0/TOTAL_N, ROC_FRR*1.0/TOTAL_P)[0] * 100
logger.info(f'Overall EER under global threshold: {final_global:.5f}')

user_all = np.array([35,35,35,35])
logger.info(f'Overall EER under user-specific threshold: {np.sum(np.array(mEER_L) * user_all) / np.sum(user_all):.5f}')