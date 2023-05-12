# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:26:07 2023

@author: xiwenc

Programming is not used in this code.

"""




## load dataset


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:32'


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms



import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
# from torch.nn import ModuleList
import torchsummary
import copy
import random

import argparse

from method_new22 import MIMO,MIMO_rand,MIMO_sample

from lib import dpp,div


from torchvision.models import resnet50, ResNet50_Weights

import math,time





plt.close('all')
    
parser = argparse.ArgumentParser()
# federated arguments

# other arguments
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--num_sample', type=int, default=500, help="number of classes")
parser.add_argument('--feature_dimension', type=int, default=300, help="number of channels of imges by ResNet50")
parser.add_argument('--max_length', type=int, default=200, help="number of classes")
parser.add_argument('--num_cluster', type=int, default=10, help="number of classes")

# parser.add_argument('--active_dim', type=int, default=20, help="GPU ID, -1 for CPU")
parser.add_argument('--verbose', default=True,type=bool, help='verbose print')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--dataset', type=str,default='CIFAR100', help='dataset name')
parser.add_argument('--num_repeat', type=int,default=10, help='number of repeat times')
# parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

T = 5


args = parser.parse_args()

Sparse = args.max_length//T//2
out_path = f'output_iid/{args.dataset}_{args.num_cluster}_T{T}_length{args.max_length}_Sparse{Sparse}/'
os.makedirs(out_path,exist_ok=True)


args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
device = args.device 


eps = 0.5 # to calculate RD-diversity

print(args)
# Feature extractor
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
model.eval()



max_length = args.max_length
# active_dim = args.active_dim



feature_dimension=args.feature_dimension
num_sample = args.num_sample
num_cluster = args.num_cluster





transform_train = transforms.Compose([
        # transforms.Scale((550, 550)),
        # transforms.RandomCrop(448, padding=8),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((256,256))
    ])


if args.dataset ==  'StanfordCars':
    
    train_data = torchvision.datasets.StanfordCars(
        root = 'data/StanfordCars',  #
        split = 'train',       #False: tEST
        # transform = torchvision.transforms.ToTensor(),
        transform=transform_train,#norm
        download=True
    )
    total_class = 196

elif args.dataset == 'CIFAR10':
     train_data = torchvision.datasets.CIFAR10(
        root = 'data/CIFAR10',  #
        train = True,       #False: tEST
        # transform = torchvision.transforms.ToTensor(),
        transform=transform_train,#norm
        download=True
    )

     total_class = 10

elif args.dataset == 'CIFAR100':
     train_data = torchvision.datasets.CIFAR100(
        root = 'data/CIFAR100',  #
        train = True,       #False: tEST
        # transform = torchvision.transforms.ToTensor(),
        transform=transform_train,#norm
        download=True
    )

     total_class = 100
    
elif args.dataset == 'Caltech256':   
     train_data = torchvision.datasets.Caltech256(
        root = 'data/Caltech256',  #
        # split = 'train',       #False: tEST
        # transform = torchvision.transforms.ToTensor(),
        transform=transform_train,#norm
        download=True
        )
     total_class = 256
    
elif args.dataset == 'GTSRB':   
     train_data = torchvision.datasets.GTSRB(
        root = 'data/GTSRB',  #
        split = 'train',       #False: tEST
        # transform = torchvision.transforms.ToTensor(),
        transform=transform_train,#norm
        download=True
        )
     total_class = 40
  

    
    
loader_all_data = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=False, num_workers=0)





for repeat_index in range(args.num_repeat):
## generate data for each source
    loader_all_data = torch.utils.data.DataLoader(
                train_data, batch_size=128, shuffle=True, num_workers=0)
        ## generate data for each source
    dataset_pre_X = torch.tensor([]) #candidate clusters data
    dataset_pre_Y = torch.tensor([])
    if  total_class == 10:
        ava_class = random.sample(range(total_class),args.num_cluster)
    elif total_class>10:
        total_sample = min(args.num_cluster*10,total_class)
        if args.dataset == 'StanfordCars':
            total_sample =  min(args.num_cluster*20,total_class)
        ava_class = random.sample(range(total_class),total_sample)
        res_class = total_sample%args.num_cluster
        if res_class!=0:
            ava_class = ava_class[:-res_class]

        ava_class = np.array(ava_class).reshape(args.num_cluster,-1)
        # print(ava_class)
    feat_cluster = [  torch.tensor([]) for i in range(args.num_cluster)]
    target_cluster = [  torch.tensor([]) for i in range(args.num_cluster)]



    feat_all = np.zeros((args.num_cluster*num_sample,feature_dimension))

    diversity = np.zeros((num_cluster))

    #generate data for each source
    
    with torch.no_grad():
        j=0
        for index_pre,(x,y) in  enumerate(loader_all_data):
            temp_feat = model(x.to(device)).detach().cpu()
            if len(target_cluster[j])< args.num_sample and j<num_cluster:
                
                feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat), 0)
                target_cluster[j] = torch.cat((target_cluster[j], y), 0)
            elif len(target_cluster[j])>= args.num_sample and j<num_cluster-1:
                j=j+1
                # print(j)
                feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat), 0)
                target_cluster[j] = torch.cat((target_cluster[j], y), 0)
            else:
                break
                
                 
            
#             for j in range(args.num_cluster):
#                 if len(target_cluster[j])< args.num_sample:
#                     if total_class == 10:
#                         c = ava_class[j]
#                         feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
#                         target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
#                     else:
#                             c_list = ava_class[j,:]
#                             for c in c_list:
#                                 feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
#                                 target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
# #                             if total_class == 10:
# #                                 c = ava_class[j]
# #                                 feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
# #                                 target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
# #                             else:
# #                                 c_list = ava_class[j,:]

# #                                 for c in c_list:
# #                                     feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
# #                                     target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
#                 else:
                    # pass

        for j in range(args.num_cluster):
            # feat_cluster[j] = feat_cluster[j][:args.num_sample].numpy()
            feat_cluster[j] = feat_cluster[j][:args.num_sample,:args.feature_dimension].numpy()
            target_cluster[j] = target_cluster[j][:args.num_sample].numpy()
            feat_all[j*num_sample:(j+1)*num_sample,:] = feat_cluster[j] 

            _,s,_ = np.linalg.svd(np.cov(feat_cluster[j].T))
            diversity[j] = sum(np.log(s[:])*feature_dimension/eps+1)
            print(feat_cluster[j].shape)
    os.makedirs(out_path+'data/',exist_ok=True)
    np.save(out_path+f'data/{np.random.randint(10000)}_data.npy',feat_all)
    
    
    
    
    
    
    
    
    ### greedy search
    
    kernel_matrix =feat_all@feat_all.T
    t = time.time()
    result_global = dpp(kernel_matrix,args.max_length )   #  max_length
    t_gloabl = time.time()-t
    div_global_list = np.zeros(T)
    for i in range(T):
        
        div_global_list[i] = div(feat_all,result_global[:(i+1)*args.max_length//T])
    
    
    ### only samples maximum diversity
    div_max_list = np.zeros(T)
    
    ix_max =np.argmax(diversity)
    subkernel =feat_all[ix_max*num_sample:(ix_max+1)*num_sample,:]@feat_all[ix_max*num_sample:(ix_max+1)*num_sample,:].T
    t = time.time()
    result_max=dpp(subkernel,args.max_length  )
    t_max = time.time()-t
    for i in range(T):
        div_max_list[i] = div(feat_all,result_max[:(i+1)*args.max_length//T])
    
    ### random slection 
    result_random = random.sample(range(len(feat_all)),args.max_length)
    div_random_list = np.zeros(T)
    for i in range(T):
        div_random_list[i] = div(feat_all,result_random[:(i+1)*args.max_length//T])
    
    ### select by our method
    # from method import MIMO,MIMO_rand
    
    active_rate = [0.,0.2,0.4,0.6,0.8,1.]
    # active_rate = [0.,1.]
    def MIMO_select(Program_flag = False,feedback_flag=True,random_flag = False):
        # active_rate = [0.]
        
        results = [ [[],[]] for i in range(len(active_rate))]
        results_select_index = [ [] for i in range(len(active_rate))]
        
        time_per = np.zeros((len(active_rate),T,3))
        results_per = np.zeros((len(active_rate),T))
        results_len =  np.zeros((len(active_rate),T))
        
        for j in range(len(active_rate)):
            try:
                # print(active_rate[j])
                count_s = 0
                all_select_index = []
                Svd_dim =  int(Sparse*active_rate[j])
                for t in range(T):
                    # print(t)
                    if t==0:
                        select_index_set = []
                        select_length = args.max_length//T
                        
                        index_temp0,(t1,t2,t3) = MIMO(select_index_set,feat_all,num_sample,select_length,num_candidate=feature_dimension,sparse=Sparse,svd_dim=Svd_dim,programming=Program_flag, feedback = False,random_weight = random_flag)
                        all_select_index.extend(index_temp0)
                        count_s = count_s+args.max_length//T
                    else:
                        select_index_set = copy.deepcopy(all_select_index)
                        if t != T-1:
                            ave_len = args.max_length//T
                            count_s = count_s+args.max_length//T
                        else:
                            ave_len = args.max_length-count_s
                        select_length = ave_len
                        index_temp,(t1,t2,t3) = MIMO(select_index_set,feat_all,num_sample,select_length,num_candidate=feature_dimension,sparse=Sparse,svd_dim=Svd_dim,programming=Program_flag, feedback = feedback_flag,random_weight = random_flag)
                
                        all_select_index.extend(index_temp)
                    results_per[j,t] = div(feat_all,all_select_index)
                    results_len[j,t] = len(all_select_index)
                    time_per[j,t,:] = [t1,t2,t3] 
                    
                if len(all_select_index)==args.max_length:
                       results[j][0].append(Sparse)
                       results[j][1].append(div(feat_all,all_select_index))
                       results_select_index[j] = all_select_index
                   
            except :
                continue  
        return results_per,results_len,results_select_index,time_per
    
    def MIMO_select_sample(Program_flag = False,feedback_flag=True,random_flag = False):
        # print(f'random_flag: {random_flag}')
        # active_rate = [0.]
        # active_rate = [0.,0.2,0.4,0.6,0.8,1.]
        results = [ [[],[]] for i in range(len(active_rate))]
        results_select_index = [ [] for i in range(len(active_rate))]
        
        
        results_per = np.zeros((len(active_rate),T))
        results_len =  np.zeros((len(active_rate),T))
        time_per = np.zeros((len(active_rate),T,3))
        
        
        for j in range(len(active_rate)):
            # print(active_rate[j])
            try:
                count_s = 0
                all_select_index = []
                Svd_dim =  int(Sparse*active_rate[j])
                for t in range(T):
                    # print(t)
                    if t==0:
                        select_index_set = []
                        select_length = args.max_length//T
                        
                        index_temp0,(t1,t2,t3) = MIMO_sample(select_index_set,feat_all,num_sample,select_length,num_candidate=feature_dimension,sparse=Sparse,svd_dim=Svd_dim,programming=Program_flag, feedback = False,random_weight = random_flag)
                        all_select_index.extend(index_temp0)
                        count_s = count_s+args.max_length//T
                    else:
                        select_index_set = copy.deepcopy(all_select_index)
                        if t != T-1:
                            ave_len = args.max_length//T
                            count_s = count_s+args.max_length//T
                        else:
                            ave_len = args.max_length-count_s
                        select_length = ave_len
                        index_temp,(t1,t2,t3) = MIMO_sample(select_index_set,feat_all,num_sample,select_length,num_candidate=feature_dimension,sparse=Sparse,svd_dim=Svd_dim,programming=Program_flag, feedback = feedback_flag,random_weight = random_flag)
                
                        all_select_index.extend(index_temp)
                    results_per[j,t] = div(feat_all,all_select_index)
                    results_len[j,t] = len(all_select_index)
                    time_per[j,t,:] = [t1,t2,t3] 
                    
                if len(all_select_index)==args.max_length:
                       results[j][0].append(Sparse)
                       results[j][1].append(div(feat_all,all_select_index))
                       results_select_index[j] = all_select_index
                       
            except :
                continue   
                   
        return results_per,results_len,results_select_index,time_per
    
    def MIMO_select_random():
        active_rate = [0.]
        results = [ [[],[]] for i in range(len(active_rate))]
        results_select_index = [ [] for i in range(len(active_rate))]
        
        
        results_per = np.zeros((len(active_rate),T))
        results_len =  np.zeros((len(active_rate),T))
        time_per = np.zeros((len(active_rate),T,3))
        for j in range(len(active_rate)):
            try:
                count_s = 0
                all_select_index = []
                Svd_dim =  int(Sparse*active_rate[j])
                for t in range(T):
                    # print(t)
                    if t==0:
                        select_index_set = []
                        select_length = args.max_length//T
                        
                        index_temp0,(t1,t2,t3) = MIMO_rand(select_index_set,feat_all,num_sample,select_length,num_candidate=feature_dimension,sparse=Sparse,svd_dim=Svd_dim,programming=False, feedback = False)
                        all_select_index.extend(index_temp0)
                        count_s = count_s+args.max_length//T
                    else:
                        select_index_set = copy.deepcopy(all_select_index)
                        if t != T-1:
                            ave_len = args.max_length//T
                            count_s = count_s+args.max_length//T
                        else:
                            ave_len = args.max_length-count_s
                        select_length = ave_len
                        index_temp,(t1,t2,t3) = MIMO_rand(select_index_set,feat_all,num_sample,select_length,num_candidate=feature_dimension,sparse=Sparse,svd_dim=Svd_dim,programming=False, feedback = True)
                
                        all_select_index.extend(index_temp)
                    results_per[j,t] = div(feat_all,all_select_index)
                    results_len[j,t] = len(all_select_index)
                    time_per[j,t,:] = [t1,t2,t3] 
                    
                if len(all_select_index)==args.max_length:
                       results[j][0].append(Sparse)
                       results[j][1].append(div(feat_all,all_select_index))
                       results_select_index[j] = all_select_index
                   
            except :
                continue       
                 
        return results_per,results_len,results_select_index,time_per
    
    
    
    # SET REPEAT FOR RANDOMW SELECTION
    step = 10
     # Nothing
    results_per_nothing_all = np.zeros(step,dtype = object)
    # for i in range(10):
    #     try:
    #         results_per1,results_len1,_ = MIMO_select()
    #         results_per_nothing_all[i] = results_per1
    #     except :
    #         continue   
    




    print(' random weight MAP H')
    results_per_randw_MAP_f_all = np.zeros((step,2),dtype = object)
    if True:
        for i in range(step):
            try:
                results_per1,results_len1,_,time_per = MIMO_select(Program_flag = False,feedback_flag=True,random_flag=True)
                results_per_randw_MAP_f_all[i,0] = results_per1
                results_per_randw_MAP_f_all[i,1] = time_per
            except :
                continue

    
    # ours
    
    print('our')
    results_per_our_all = np.zeros((step,2),dtype = object)
    
    
    
    if True:
        for i in range(step):
            try:
                results_per1,results_len1,_,time_per = MIMO_select(Program_flag = False,feedback_flag=True)
                results_per_our_all[i,0] = results_per1
                results_per_our_all[i,1] = time_per
            except :
                continue
        # print(results_len1)
    # uniform no feedback
    print('uniform no feedback')
    results_per_uniform_no_f_all = np.zeros((1,2),dtype = object)
    if True:
        for i in range(1):
            try:
                results_per2,results_len1,_,time_per = MIMO_select(Program_flag = False,feedback_flag=False)
                results_per_uniform_no_f_all[i,0] = results_per2
                results_per_uniform_no_f_all[i,1] = time_per
            except :
                continue
    # print(results_len1)
    # no programing sampling-based
    print(' no programing, sampling-based, uniform')
    results_per_sampling_f_all = np.zeros((step,2),dtype = object)
    if True:
        for i in range(step):
            try:
                results_per1,results_len1,_,time_per = MIMO_select_sample(Program_flag = False,feedback_flag=True)
                results_per_sampling_f_all[i,0] = results_per1
                results_per_sampling_f_all[i,1] = time_per
            except :
                continue
            
    # 
    print('random submatrix with uniform power')
    results_random_all = np.zeros((10,2),dtype = object)
    for i in range(10):
        results_per1,results_len1,_,time_per = MIMO_select_random()
        results_random_all[i,0] = results_per1
        results_random_all[i,1] = time_per
    
    
    
    
    
    
    print(' no programing sampling-based, ranodm wight')
    results_per_randw_sampling_f_all = np.zeros((step,2),dtype = object)
    if True:
        for i in range(step):
            try:
                results_per1,results_len1,_,time_per = MIMO_select_sample(Program_flag = False,feedback_flag=True,random_flag=True)
                results_per_randw_sampling_f_all[i,0] = results_per1
                results_per_randw_sampling_f_all[i,1] = time_per
            except :
                continue
    
    
    all_resluts = np.array([ div_global_list,
                            div_max_list,
                            results_per_nothing_all,
                            results_per_our_all,
                            results_per_uniform_no_f_all, results_per_sampling_f_all,
                            results_random_all,
                            results_per_randw_MAP_f_all,
                            results_per_randw_sampling_f_all,
                            div_random_list,
                            t_gloabl,t_max,args],dtype=object)
    np.save(out_path+f'{np.random.randint(10000)}.npy',all_resluts)
