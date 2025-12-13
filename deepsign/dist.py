# -*- coding:utf-8 -*-

import numpy as np
from fastdtw import fastdtw

def dist_seq(features,feat_freq,template_num,nf,genuine_num,forgery_num):
    dist_positive,dist_negative,dist_template = [],[],[]
    dist_fpositive,dist_fnegative,dist_ftemplate = [],[],[]
    for idx,feat in enumerate(features):
        feat_a = feat[0:template_num]
        feat_p = feat[(template_num + nf):(genuine_num + nf)]
        feat_n = feat[(genuine_num + nf):]

        feat_fa = feat_freq[idx][0:template_num]
        feat_fp = feat_freq[idx][(template_num + nf):(genuine_num + nf)]
        feat_fn = feat_freq[idx][(genuine_num + nf):]
        
        dist_p = np.zeros((genuine_num - template_num,template_num))
        dist_n = np.zeros((forgery_num,template_num))
        dist_t = np.zeros((template_num,template_num))
        dist_fp = np.zeros((genuine_num - template_num,template_num))
        dist_fn = np.zeros((forgery_num,template_num))
        dist_ft = np.zeros((template_num,template_num))
        for i in range(template_num):
            fa1 = feat_a[i]
            fas = np.sum(fa1,axis=1)
            fa1 = np.delete(fa1,np.where(fas == 0)[0],axis=0)
            freq_fa1 = feat_fa[i]
            for j in range(i + 1,template_num):
                fa2 = feat_a[j]
                fas = np.sum(fa2,axis=1)
                fa2 = np.delete(fa2,np.where(fas == 0)[0],axis=0)
                dist,_ = fastdtw(fa2,fa1,radius=2,dist=1)
                dist = dist / (fa2.shape[0] + fa1.shape[0])
                freq_fa2 = feat_fa[j]
                dist_freq = np.sqrt(np.sum(np.power(freq_fa2 - freq_fa1,2),axis=0))
                dist_t[i,j] = dist
                dist_ft[i,j] = dist_freq
                # dist_t[i,j] = adaptive_weight(dist,dist_freq)
        for i in range(genuine_num - template_num):
            fp = feat_p[i]
            fps = np.sum(fp,axis=1)
            fp = np.delete(fp,np.where(fps == 0)[0],axis=0)
            freq_fp = feat_fp[i]
            for j in range(template_num):
                fa = feat_a[j]
                fas = np.sum(fa,axis=1)
                fa = np.delete(fa,np.where(fas == 0)[0],axis=0)
                dist,_ = fastdtw(fp,fa,radius=2,dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                freq_fa = feat_fa[j]
                dist_freq = np.sqrt(np.sum(np.power(freq_fp - freq_fa,2),axis=0))
                dist_p[i,j] = dist
                dist_fp[i,j] = dist_freq
                # dist_p[i,j] = adaptive_weight(dist,dist_freq)
        for i in range(forgery_num):
            fn = feat_n[i]
            fns = np.sum(fn,axis=1)
            fn = np.delete(fn,np.where(fns == 0)[0],axis=0)
            freq_fn = feat_fn[i]
            for j in range(template_num):
                fa = feat_a[j]
                fas = np.sum(fa,axis=1)
                fa = np.delete(fa,np.where(fas == 0)[0],axis=0)
                dist,_ = fastdtw(fn,fa,radius=2,dist=1)
                dist = dist / (fn.shape[0] + fa.shape[0])
                freq_fa = feat_fa[j]
                dist_freq = np.sqrt(np.sum(np.power(freq_fn - freq_fa,2),axis=0))
                dist_n[i,j] = dist
                dist_fn[i,j] = dist_freq
                # dist_n[i,j] = adaptive_weight(dist,dist_freq)
        dist_positive.append(dist_p)
        dist_negative.append(dist_n)
        dist_template.append(dist_t)
        dist_fpositive.append(dist_fp)
        dist_fnegative.append(dist_fn)
        dist_ftemplate.append(dist_ft)
    dist_positive = np.concatenate(dist_positive,axis=0)
    dist_negative = np.concatenate(dist_negative,axis=0)
    dist_template = np.concatenate(dist_template,axis=0)
    dist_fpositive = np.concatenate(dist_fpositive,axis=0)
    dist_fnegative = np.concatenate(dist_fnegative,axis=0)
    dist_ftemplate = np.concatenate(dist_ftemplate,axis=0)
    return dist_positive,dist_negative,dist_template,dist_fpositive,dist_fnegative,dist_ftemplate

def dist_seq_rf(features,feat_freq,template_num,nf,genuine_num,forgery_num):
    dist_positive,dist_negative,dist_template = [],[],[]
    dist_fpositive,dist_fnegative,dist_ftemplate = [],[],[]
    features_anchor,features_positive = [],[]
    freq_features_anchor,freq_features_positive = [],[]
    for i,feat in enumerate(features):
        feat_a = feat[0:template_num]
        feat_p = feat[(template_num + nf):(genuine_num + nf)]

        features_anchor.append(feat_a)
        features_positive.append(feat_p)

        feat_fa = feat_freq[i][0:template_num]
        feat_fp = feat_freq[i][(template_num + nf):(genuine_num + nf)]
        freq_features_anchor.append(feat_fa)
        freq_features_positive.append(feat_fp)
    for i,feat_a in enumerate(features_anchor):
        feat_p = features_positive[i]
        feat_fa = freq_features_anchor[i]
        feat_fp = freq_features_positive[i]
        feat_n,freq_feat_n = [],[]
        for j in range(len(features_anchor)):
            if i != j:
                feat_n.append(features_positive[j][2])
                freq_feat_n.append(freq_features_positive[j][2])
        dist_p = np.zeros((len(feat_p),template_num))
        dist_n = np.zeros((len(feat_n),template_num))
        dist_t = np.zeros((template_num,template_num))
        dist_fp = np.zeros((len(feat_p),template_num))
        dist_fn = np.zeros((len(feat_n),template_num))
        dist_ft = np.zeros((template_num,template_num))
        for j in range(template_num):
            fa1 = feat_a[j]
            fas = np.sum(fa1,axis=1)
            fa1 = np.delete(fa1,np.where(fas == 0)[0],axis=0)
            freq_fa1 = freq_features_anchor[i][j]
            for k in range(j + 1,template_num):
                fa2 = feat_a[k]
                fas = np.sum(fa2,axis=1)
                fa2 = np.delete(fa2,np.where(fas == 0)[0],axis=0)
                dist,_ = fastdtw(fa2,fa1,radius=2,dist=1)
                dist = dist / (fa2.shape[0] + fa1.shape[0])
                freq_fa2 = feat_fa[k]
                dist_freq = np.sqrt(np.sum(np.power(freq_fa2 - freq_fa1,2),axis=0))
                dist_t[j,k] = dist
                dist_ft[j,k] = dist_freq
        for j in range(len(feat_p)):
            fp = feat_p[j]
            fps = np.sum(fp,axis=1)
            fp = np.delete(fp,np.where(fps == 0)[0],axis=0)
            freq_fp = feat_fp[j]
            for k in range(template_num):
                fa = feat_a[k]
                fas = np.sum(fa,axis=1)
                fa = np.delete(fa,np.where(fas == 0)[0],axis=0)
                dist,_ = fastdtw(fp,fa,radius=2,dist=1)
                dist = dist / (fp.shape[0] + fa.shape[0])
                freq_fa = feat_fa[k]
                dist_freq = np.sqrt(np.sum(np.power(freq_fp - freq_fa,2),axis=0))
                dist_p[j,k] = dist
                dist_fp[j,k] = dist_freq
        for j in range(len(feat_n)):
            fn = feat_n[j]
            fns = np.sum(fn,axis=1)
            fn = np.delete(fn,np.where(fns == 0)[0],axis=0)
            freq_fn = freq_feat_n[j]
            for k in range(template_num):
                fa = feat_a[k]
                fas = np.sum(fa,axis=1)
                fa = np.delete(fa,np.where(fas == 0)[0],axis=0)
                dist,_ = fastdtw(fn,fa,radius=2,dist=1)
                dist = dist / (fn.shape[0] + fa.shape[0])
                freq_fa = feat_fa[k]
                dist_freq = np.sqrt(np.sum(np.power(freq_fn - freq_fa,2),axis=0))
                dist_n[j,k] = dist
                dist_fn[j,k] = dist_freq
        dist_positive.append(dist_p)
        dist_negative.append(dist_n)
        dist_template.append(dist_t)
        dist_fpositive.append(dist_fp)
        dist_fnegative.append(dist_fn)
        dist_ftemplate.append(dist_ft)
    dist_positive = np.concatenate(dist_positive,axis=0)
    dist_negative = np.concatenate(dist_negative,axis=0)
    dist_template = np.concatenate(dist_template,axis=0)
    dist_fpositive = np.concatenate(dist_fpositive,axis=0)
    dist_fnegative = np.concatenate(dist_fnegative,axis=0)
    dist_ftemplate = np.concatenate(dist_ftemplate,axis=0)
    return dist_positive,dist_negative,dist_template,dist_fpositive,dist_fnegative,dist_ftemplate