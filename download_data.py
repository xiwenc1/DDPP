
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

from method import MIMO,MIMO_rand

from lib import dpp,div


from torchvision.models import resnet50, ResNet50_Weights
import time






plt.close('all')
    
parser = argparse.ArgumentParser()
# federated arguments

# other arguments
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('--num_sample', type=int, default=500, help="number of classes")
parser.add_argument('--feature_dimension', type=int, default=1000, help="number of channels of imges by ResNet50")
parser.add_argument('--max_length', type=int, default=250, help="number of classes")
parser.add_argument('--num_cluster', type=int, default=10, help="number of classes")

# parser.add_argument('--active_dim', type=int, default=20, help="GPU ID, -1 for CPU")
parser.add_argument('--verbose', default=True,type=bool, help='verbose print')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--dataset', type=str,default='CIFAR10', help='dataset name')
parser.add_argument('--num_repeat', type=int,default=1, help='number of repeat times')
# parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

T = 5


args = parser.parse_args()

Sparse = args.max_length//T
out_path = f'output/{args.dataset}_{args.num_cluster}_T{T}_length{args.max_length}_Sparse{Sparse}/'
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


for dataset in  ['StanfordCars','CIFAR10','CIFAR100','GTSRB']: #, Caltech256 ['StanfordCars','CIFAR10','CIFAR100','GTSRB']:
# for dataset in ['GTSRB']:
    
    args.dataset = dataset 

    transform_train = transforms.Compose([
            # transforms.Scale((550, 550)),
            # transforms.RandomCrop(448, padding=8),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((256,256 )),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            
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



    # print(train_data.targets)
    
    
    


    print(f'{dataset} has total class {total_class} for {args.num_cluster} clusters')


    for repeat_index in range(args.num_repeat):
        t = time.time()
    
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
            for index_pre,(x,y) in  enumerate(loader_all_data):

                temp_feat = model(x.to(device)).detach().cpu()
                for j in range(args.num_cluster):
                    if len(target_cluster[j])< args.num_sample:
                        if total_class == 10:
                            c = ava_class[j]
                            feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
                            target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
                        else:
                                c_list = ava_class[j,:]
                                for c in c_list:
                                    feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
                                    target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
#                             if total_class == 10:
#                                 c = ava_class[j]
#                                 feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
#                                 target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
#                             else:
#                                 c_list = ava_class[j,:]

#                                 for c in c_list:
#                                     feat_cluster[j] = torch.cat((feat_cluster[j], temp_feat[y==c]), 0)
#                                     target_cluster[j] = torch.cat((target_cluster[j], y[y==c]), 0)
                    else:
                        pass

            for j in range(args.num_cluster):
                feat_cluster[j] = feat_cluster[j][:args.num_sample].numpy()
                target_cluster[j] = target_cluster[j][:args.num_sample].numpy()
                feat_all[j*num_sample:(j+1)*num_sample,:] = feat_cluster[j] 

                _,s,_ = np.linalg.svd(np.cov(feat_cluster[j].T))
                diversity[j] = sum(np.log(s[:])*feature_dimension/eps+1)
                print(feat_cluster[j].shape)
            
            
            
        print(f'{dataset} sccussful. Use {time.time()-t} seconds')
    