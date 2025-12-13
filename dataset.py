# -*- coding:utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
from utils import extract_temporal_features

class Numbers(Dataset):
    def __init__(self,handwriting_info,transform=None,train=True):
        super().__init__()
        self.users = handwriting_info.keys()
        self.users_cnt = len(self.users)
        self.train = train
        self.features = []
        self.genuine_cnt = np.zeros(self.users_cnt,dtype=np.int32)
        self.forgery_cnt = np.zeros(self.users_cnt,dtype=np.int32)
        self.session_labels = []
        for i,k in enumerate(self.users):
            extract_temporal_features(handwriting_info[k]['genuine'],self.features)
            extract_temporal_features(handwriting_info[k]['forgery'],self.features)
            self.genuine_cnt[i] = len(handwriting_info[k]['genuine'])
            self.forgery_cnt[i] = len(handwriting_info[k]['forgery'])
        self.features_cnt = len(self.features)
        accu_indices = np.cumsum(self.genuine_cnt + self.forgery_cnt)
        self.accu_indices = np.roll(accu_indices,1)
        self.accu_indices[0] = 0
        self.feature_dims = np.shape(self.features[0])[1]
        user_interval = self.genuine_cnt[0] + self.forgery_cnt[0]
        self.user_labels = []
        for i in range(self.users_cnt):
            self.user_labels.extend([i] * user_interval)
        self.binary_labels = ([1] * self.genuine_cnt[0] + [0] * self.forgery_cnt[0]) * self.users_cnt
        self.transform = transform

    @property
    def config(self):
        config_info = {
            'users_cnt':self.users_cnt,
            'accu_indices':self.accu_indices,
            'forgery_cnt':self.forgery_cnt,
            'genuine_cnt':self.genuine_cnt,
            'features_cnt':self.features_cnt,
        }
        return config_info

    def __len__(self):
        return self.features_cnt

    def __getitem__(self,idx):
        if self.train:
            feature = self.features[idx]
            if self.transform is not None:
                feature = self.transform(feature)
        else:
            feature = self.features[idx]
        return feature,len(feature),self.user_labels[idx],self.binary_labels[idx]

class DeepSignDB(Dataset):
    def __init__(self,handwriting_info,transform=None,train=True,finger_mode=False):
        super().__init__()
        self.users = handwriting_info.keys()
        self.users_cnt = len(self.users)
        self.train = train
        self.features = []
        self.genuine_cnt = np.zeros(self.users_cnt,dtype=np.int32)
        self.forgery_cnt = np.zeros(self.users_cnt,dtype=np.int32)
        self.binary_labels = []
        for i,k in enumerate(self.users):
            extract_temporal_features(handwriting_info[k][True],self.features,finger_mode=finger_mode)
            extract_temporal_features(handwriting_info[k][False],self.features,finger_mode=finger_mode)
            self.genuine_cnt[i] = len(handwriting_info[k][True])
            self.forgery_cnt[i] = len(handwriting_info[k][False])
            self.binary_labels.extend([1] * self.genuine_cnt[i] + [0] * self.forgery_cnt[i])
        self.features_cnt = len(self.features)
        accu_indices = np.cumsum(self.genuine_cnt + self.forgery_cnt)
        self.accu_indices = np.roll(accu_indices,1)
        self.accu_indices[0] = 0
        self.feature_dims = np.shape(self.features[0])[1]
        self.transform = transform
        self.user_labels = []
        user_interval = self.genuine_cnt[0] + self.forgery_cnt[0]
        for i in range(self.users_cnt):
            self.user_labels.extend([i] * user_interval)

    @property
    def config(self):
        config_info = {
            'users_cnt':self.users_cnt,
            'accu_indices':self.accu_indices,
            'forgery_cnt':self.forgery_cnt,
            'genuine_cnt':self.genuine_cnt,
            'features_cnt':self.features_cnt,
        }
        return config_info

    def __len__(self):
        return self.features_cnt

    def __getitem__(self,idx):
        feature = self.features[idx]
        if self.train:
            if self.transform is not None:
                feature = self.transform(feature)
        return feature,len(feature),self.user_labels[idx],self.binary_labels[idx]

    def add_data(self,handwriting_info,finger_mode=False):
        cur_users = handwriting_info.keys()
        cur_users_cnt = len(handwriting_info)
        genuine_cnt = np.zeros(cur_users_cnt,dtype=np.int32)
        forgery_cnt = np.zeros(cur_users_cnt,dtype=np.int32)
        for i,k in enumerate(cur_users):
            extract_temporal_features(handwriting_info[k][True],self.features,finger_mode=finger_mode)
            extract_temporal_features(handwriting_info[k][False],self.features,finger_mode=finger_mode)
            genuine_cnt[i] = len(handwriting_info[k][True])
            forgery_cnt[i] = len(handwriting_info[k][False])
            self.binary_labels.extend([1] * genuine_cnt[i] + [0] * forgery_cnt[i])
        self.features_cnt = len(self.features)
        self.forgery_cnt = np.concatenate((self.forgery_cnt,forgery_cnt))
        self.genuine_cnt = np.concatenate((self.genuine_cnt,genuine_cnt))
        accu_indices = np.cumsum(self.genuine_cnt + self.forgery_cnt)
        self.accu_indices = np.roll(accu_indices,1)
        self.accu_indices[0] = 0
        last_users_cnt = self.users_cnt
        self.users_cnt += cur_users_cnt
        user_interval = genuine_cnt[0] + forgery_cnt[0]
        for i in range(last_users_cnt,self.users_cnt):
            self.user_labels.extend([i] * user_interval)

class TrainSampler:
    def __init__(self,dataset_config:dict,user_sample,genuine_sample=5,forgery_sample=5):
        self.__dict__.update(dataset_config)
        self.users_indices = np.arange(0,self.users_cnt,dtype=np.int32)
        self.user_sample = user_sample
        self.genuine_sample = genuine_sample
        self.forgery_sample = forgery_sample

    def __len__(self):
        return self.users_cnt

    def __iter__(self):
        np.random.shuffle(self.users_indices)
        for _ in range(self.users_cnt):
            batch_indices = []
            indices = np.random.choice(self.users_indices,size=self.user_sample,replace=False)
            for idx in indices:
                genuine_indices = np.random.choice(np.arange(0,self.genuine_cnt[idx]),
                    size=self.genuine_sample,replace=False)
                forgery_indices = np.random.choice(np.arange(0,self.forgery_cnt[idx]),
                    size=self.forgery_sample // 2,replace=False) + self.genuine_cnt[idx]
                batch_indices.append(genuine_indices + self.accu_indices[idx])
                batch_indices.append(forgery_indices + self.accu_indices[idx])
                random_forgery_users = (idx + np.random.randint(1,self.users_cnt - 1,size=self.forgery_sample // 2)) % self.users_cnt
                for rf_user in random_forgery_users:
                    rf_idx = np.random.choice(self.genuine_cnt[rf_user],size=1,replace=False)
                    batch_indices.append(rf_idx + self.accu_indices[rf_user])
            batch_indices = np.concatenate(batch_indices,axis=0).astype(np.int32)
            yield batch_indices

class TestSampler:
    def __init__(self,users_cnt,genuine_sample=5,forgery_sample=5):
        self.users_cnt = users_cnt
        self.users_indices = np.arange(0,self.users_cnt,dtype=np.int32)
        self.user_sample = genuine_sample + forgery_sample
        self.genuine_sample = genuine_sample
        self.forgery_sample = forgery_sample

    def __len__(self):
        return self.users_cnt

    def __iter__(self):
        for i in range(self.users_cnt):
            batch_indices = np.arange(i * self.user_sample,(i + 1) * self.user_sample,dtype=np.int32)
            yield batch_indices
   
class TestSamplerDeepSign:
    def __init__(self,dataset_config):
        self.__dict__.update(dataset_config)
        self.users_indices = np.arange(0,self.users_cnt,dtype=np.int32)
        self.user_sample = self.genuine_cnt[0] + self.forgery_cnt[0] # 每个batch的用户个数

    def __len__(self):
        return self.users_cnt

    def __iter__(self):
        # np.random.shuffle(self.users_indices)
        for i in range(self.users_cnt): # 每次都取一个用户，每个都拿全部20个，然后55个就取完了
            batch_indices = np.arange(i * self.user_sample,(i + 1) * self.user_sample,dtype=np.int32)
            # print(batch_indices)
            yield batch_indices

def collate_fn(batch:list):
    # batch: 列表，第一个是笔迹信息，第二个是长度
    batch_size = len(batch)
    handwriting = [i[0] for i in batch]
    hw_len = np.array([i[1] for i in batch],dtype=np.float32)
    user_labels = np.array([i[2] for i in batch])
    binary_labels = np.array([i[3] for i in batch])
    max_len = int(np.max(hw_len))
    time_function_cnts = np.shape(handwriting[0])[1]
    handwriting_padded = np.zeros((batch_size,max_len,time_function_cnts),dtype=np.float32)
    for i,hw in enumerate(handwriting):
        handwriting_padded[i,:hw.shape[0]] = hw
    return handwriting_padded,hw_len,user_labels,binary_labels