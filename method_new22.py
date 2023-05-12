# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:50:08 2023

@author: xiwenc
"""

import numpy as np
import copy
import math,time



from torchvision import datasets, transforms
import torch


import torch.nn.functional as F

import matplotlib.pyplot as plt

import random

from lib import dpp,dpp_min,dpp_d
from dppy.finite_dpps import FiniteDPP


def MIMO(select_index_set,feat_all,num_sample,select_length,num_candidate=100,sparse=1,svd_dim=0,programming=False, feedback = True,random_weight=False): 

    # print(programming)
    print(random_weight)
    dim = feat_all.shape[1]
    dim_dpp= int(np.sqrt((sparse-svd_dim)*dim*2))

    res_pxiel =  sparse*dim-  dim_dpp**2-svd_dim*dim
    
    
    
    # dim_dpp= 100

    # res_pxiel = 0
    
    epsilon = 2000
    
    # net_glob = net_glob.to(device)
    # net_glob.eval()
    
    num_cluster = len(feat_all)//num_sample
    
    feat_all_temp = copy.deepcopy(feat_all)
    index_all = []


    index_candidate = -np.ones((num_cluster,num_candidate),dtype=int)
    
    
    function_score = -np.ones((num_cluster,num_candidate),dtype=float)
    function_score2 = -np.ones((num_cluster,num_candidate),dtype=float)
    
    d_all = []# np.zeros((num_cluster,num_candidate))
    
    index_all_clsuter = -np.ones((num_cluster,num_candidate),dtype=int)   #record slected index for each ClUSTER
    
    # index_all_clsuter = [[]*num_cluster]

    index_all_candidate = []

    density = np.zeros((num_cluster))


    select_cluster_index = np.array(select_index_set)//num_sample
    
    
    
    if select_index_set==[]:
        feedback = False
        
    if feedback == False:
        message = np.eye(dim)
        Message_all = message
        
        t1=0
        t2=0
    else:
        pass
            
   #generate candidate
    for i in range(num_cluster):
        select_cluster_index_i = np.where(select_cluster_index==i)
        cluster_index = np.array((range(i*num_sample,i*num_sample+num_sample)))
        select_cluster = set(np.array(select_index_set)[select_cluster_index_i])
        ave_index = set(cluster_index) 
        ave_index = np.array(list(ave_index))
        
        num_candidate_temp = min(num_candidate,len(ave_index))
        
        
        if feedback == True:
            message_index = list(set(np.array(select_index_set))-set(select_cluster))
                        
            feat_all_temp = copy.deepcopy(feat_all)
            
            t = time.time()
            pre_code = np.eye(len(feat_all_temp[message_index,:].T))-feat_all_temp[message_index,:].T@np.linalg.inv(feat_all_temp[message_index,:]@feat_all_temp[message_index,:].T)@feat_all_temp[message_index,:]
            
            
            # DPP = FiniteDPP('likelihood', **{'L': pre_code} )
            
            active_dim = []
            
            # while len(active_dim)<dim_dpp:
            #     DPP.sample_exact_k_dpp(size=select_length)
            #     active_dim = []
            #     for rlist in DPP.list_of_samples:
            #         active_dim.extend(rlist)
                
            #     active_dim = list(set(active_dim))
                
            # active_dim = active_dim[:dim_dpp]
            # dim_kernel = feat_all_temp[select_index_set,:].T@feat_all_temp[select_index_set,:]
            active_dim = dpp(pre_code,dim_dpp)
            
            if dim_dpp==0:
                t1=0
            else:
                t1 = time.time()-t
        
            if len(active_dim)<dim_dpp:
                print(f' active {len(active_dim)}   dim_dpp{dim_dpp}')
                print('error')
                res_pxiel =  sparse*dim-  len(active_dim)**2-svd_dim*dim
                # return
        
            message = np.zeros((dim,dim))
                        
            count_a =0
            if dim_dpp>0:
                for m1 in range(dim):
                    for m2 in range(dim):
                        if (m1 in active_dim) and (m2 in active_dim):
                            message[m1,m2] = pre_code[m1,m2]
                            count_a = count_a+1
        
            count_b =0
            
            
            # m1=np.where(np.diag(message)==0)[0]
            
            # message[m1,m1] = message[m1,m1]
            # count_b = count_b+len(m1)
            while count_b<res_pxiel:
                m1 = random.sample(range(dim),1)
                m2 = random.sample(range(dim),1)
                if (m1 not in active_dim) and (m2 not in active_dim):
                    count_b= count_b+1
                    message[m1,m2] = pre_code[m1,m2]
                    message[m2,m1] = pre_code[m2,m1]
            # print(f'count{count_b}')
            
            
            
            
            
            
            
            
            t =  time.time()
            
            if svd_dim!=0:
                # print('here')
                precode_rest = pre_code-message
                Uma,Sma,_ = np.linalg.svd(precode_rest)
            
            
                message2 = Uma[:,:svd_dim]@np.diag(Sma[:svd_dim])@Uma[:,:svd_dim].T
                Message_all = message2+message
            else:
                Message_all = message
                
            t2 = time.time()-t
        
        
        
        
        
        
        
        
        
       
    
        feat_temp = copy.deepcopy(feat_all_temp)[ave_index,:]
        if feedback == False:
            subkernel =  feat_temp@feat_temp.T
        else:
            
            U1,S1,_ = np.linalg.svd(Message_all)
            H= U1@np.sqrt(np.diag(S1))@U1.T
            # H = 
            
            alpha = 1
            
            feat_temp_map = feat_temp@(alpha*H+np.eye(dim))
            # feat_temp_map = feat_temp@(alpha*H)
            # subkernel = feat_temp@Message_all@feat_temp.T+ feat_temp@feat_temp.T
            subkernel =feat_temp_map@feat_temp_map.T
        t = time.time()
        index_select,d_list = dpp_d(subkernel,num_candidate_temp)
        # print(d_list)
        index_select = np.array(ave_index)[index_select]
        t3 = time.time()-t
        for dup_indx in select_cluster:
            index_select=np.delete(index_select,np.where(index_select==dup_indx))
        # print(len(index_select))
        # index_candidate[i,:len(index_select)]=index_select
        index_candidate[i,:len(index_select)]=index_select
        
        # t = time.time()
        # index_select = dpp(subkernel,num_candidate_temp)
        # index_select = np.array(ave_index)[index_select]
        # t3 = time.time()-t
        # index_candidate[i,:len(index_select)]=index_select
        
        
        
        len_all = int( select_length)
        
        
        #calcluate the score function and find the optimized weight
        if num_cluster==1:
            # return index_candidate[i,:len_all],(t1,t2,t3)
            pass
        else:
            if programming==True:
                # print('here')
                for j in range(len(index_select)):
                    index_now = index_select[:j+1]
                    # feat_var = np.var(feat_all_temp[index_now,:],axis=0)
                    # feat_mean = (feat_all_temp[index_now,:]-np.mean(feat_all_temp[index_now,:],axis=0))/(j+1)
                    # cov = feat_mean@feat_mean.T
                    # cov = np.cov(index_now.T)
                    # _,s_now,_ = np.linalg.svd(cov)
                    # cov_now = (feat_all_temp[index_now,:]@feat_all_temp[index_now,:].T)/(j+1)
                    # cov_now = (feat_all_temp[index_now,:]@feat_all_temp[index_now,:].T)
                    # _,s_now,_ = np.linalg.svd(feat_all_temp[index_now,:])
                    # function_score[i,j] =(sum(np.log(epsilon*s_now+1/len_all))+ (dim-j-1)* np.log(1/len_all))*(j+1)/len_all
                    # function_score[i,j] =sum(np.log(epsilon*s_now+1))*(j+1)
                    # function_score[i,j] =sum(np.log(epsilon*feat_var+1))
                    function_score[i,j] =sum(np.log(d_list[:j+1]))
                
                plt.figure()
                plt.plot(function_score[i,:])
    if programming==True:
        # find the weight
        Z_size = [len_all for _ in range(num_cluster)]
        Z = np.zeros(Z_size)
        
        for index, value in np.ndenumerate(Z):
            if sum(index)+num_cluster== len_all:
                for K in range(num_cluster):
                    Z[index] = Z[index]+ function_score[i,index[K]]     #
        if num_cluster==2:
            plt.figure()
            plt.imshow(Z)
            plt.colorbar()
            plt.show()
        

        opt = np.array(np.where(Z==np.nanmax(Z)))
        print(opt)
        opt_num = opt.shape[1]
        
        Weight = (opt[:,random.sample(range(opt_num),1)].reshape(-1)+np.ones(num_cluster))
                
                
            
    else:
        Weight=np.ones(num_cluster)
            
            
    if random_weight== True:
        Weight = np.random.rand(num_cluster)        
        
    Weight =  Weight/sum(Weight)
    print(Weight)
    
    len_avai = len_all
   
    uniform_index = []

    for i in range(num_cluster):
        
        if i!= num_cluster-1:
            len_a = int(len_all*Weight[i])
            len_avai = len_avai-len_a
        else:
            len_a = len_avai
            
        if len_a>0:
            uniform_index.extend( list(index_candidate[i,:len_a]))
    
    uniform_index = np.array(uniform_index)
    # feat  = feat_all_temp[uniform_index,:]
    # subkernel =  feat@feat.T
    
    # uniform_index_order,dd = dpp_d(subkernel,max_length)
    # uniform_index_order = uniform_index[uniform_index_order]
    
    return uniform_index,(t1,t2,t3) #uniform_index_order
   
    
def MIMO_sample(select_index_set,feat_all,num_sample,select_length,num_candidate=100,sparse=1,svd_dim=0,programming=False, feedback = True, random_weight = False): 

    # print(programming)
    dim = feat_all.shape[1]
    dim_dpp= int(np.sqrt((sparse-svd_dim)*dim*2))

    res_pxiel =  sparse*dim-  dim_dpp**2-svd_dim*dim
    
    
    
    # dim_dpp= 100

    # res_pxiel = 0
    
    epsilon = 2000
    
    # net_glob = net_glob.to(device)
    # net_glob.eval()
    
    num_cluster = len(feat_all)//num_sample
    
    feat_all_temp = copy.deepcopy(feat_all)
    index_all = []


    index_candidate = -np.ones((num_cluster,num_candidate),dtype=int)
    
    
    function_score = -np.ones((num_cluster,num_candidate),dtype=float)
    function_score2 = -np.ones((num_cluster,num_candidate),dtype=float)
    
    d_all = []# np.zeros((num_cluster,num_candidate))
    
    index_all_clsuter = -np.ones((num_cluster,num_candidate),dtype=int)   #record slected index for each ClUSTER
    
    # index_all_clsuter = [[]*num_cluster]

    index_all_candidate = []

    density = np.zeros((num_cluster))


    select_cluster_index = np.array(select_index_set)//num_sample
    
    
    
    if select_index_set==[]:
        feedback = False
        
    if feedback == False:
        message = np.eye(dim)
        Message_all = message
        t1=0
        t2=0
    else:
        pass
            
   #generate candidate
    for i in range(num_cluster):
        select_cluster_index_i = np.where(select_cluster_index==i)
        cluster_index = np.array((range(i*num_sample,i*num_sample+num_sample)))
        select_cluster = set(np.array(select_index_set)[select_cluster_index_i])
        # ave_index = set(cluster_index) - set(np.array(select_index_set)[select_cluster_index_i])
        ave_index = set(cluster_index) 
        ave_index = np.array(list(ave_index))
        
        
        # message_index = select_index_set
        message_index = list(set(np.array(select_index_set))-set(select_cluster))
        feat_all_temp = copy.deepcopy(feat_all)
        
        t = time.time()
        pre_code = np.eye(len(feat_all_temp[message_index,:].T))-feat_all_temp[message_index,:].T@np.linalg.inv(feat_all_temp[message_index,:]@feat_all_temp[message_index,:].T)@feat_all_temp[message_index,:]
        
        # dim_kernel = feat_all_temp[select_index_set,:].T@feat_all_temp[select_index_set,:]
        # active_dim = dpp(dim_kernel,dim_dpp)
        
        DPP = FiniteDPP('likelihood', **{'L': pre_code} )
        
        active_dim = []
        
        while len(active_dim)<dim_dpp:
            DPP.sample_exact_k_dpp(size=select_length)
            active_dim = []
            for rlist in DPP.list_of_samples:
                active_dim.extend(rlist)
            
            active_dim = list(set(active_dim))
            
        active_dim = active_dim[:dim_dpp]
            
        # active_dim = dpp(pre_code,dim_dpp)
        if dim_dpp==0:
            t1=0
        else:
            t1 = time.time()-t
    
        if len(active_dim)<dim_dpp:
            print('error')
            res_pxiel =  sparse*dim-  len(active_dim)**2-svd_dim*dim
            # return
    
        message = np.zeros((dim,dim))
                    
        count_a =0
        if dim_dpp>0:
            for m1 in range(dim):
                for m2 in range(dim):
                    if (m1 in active_dim) and (m2 in active_dim):
                        message[m1,m2] = pre_code[m1,m2]
                        count_a = count_a+1
    
        count_b =0
        # m1=np.where(np.diag(message)==0)[0]
        # print(len(m1))
        # message[m1,m1] = message[m1,m1]
        # count_b = count_b+len(m1)
        while count_b<res_pxiel:
            m1 = random.sample(range(dim),1)
            m2 = random.sample(range(dim),1)
            if (m1 not in active_dim) and (m2 not in active_dim):
                count_b= count_b+1
                message[m1,m2] = pre_code[m1,m2]
                message[m2,m1] = pre_code[m2,m1]
        # while count_b<res_pxiel:
            
            
        #     m1 = random.sample(range(dim),1)
        #     m2 = random.sample(range(dim),1)
        #     if (m1 not in active_dim) and (m2 not in active_dim):
        #         count_b= count_b+1
        #         message[m1,m2] = pre_code[m1,m2]
                
        # print(f'count{count_b}')
        
        
        
        
        
        t =  time.time()
        
        if svd_dim!=0:
            # print('here')
            precode_rest = pre_code-message
            Uma,Sma,_ = np.linalg.svd(precode_rest)
        
        
            message2 = Uma[:,:svd_dim]@np.diag(Sma[:svd_dim])@Uma[:,:svd_dim].T
            Message_all = message2+message
            # t2 = time.time()-t
        else:
            Message_all = message
            
        t2 = time.time()-t
        
        num_candidate_temp = min(num_candidate,len(ave_index))
    
        feat_temp = copy.deepcopy(feat_all_temp)[ave_index,:]
        if feedback == False:
            subkernel =  feat_temp@feat_temp.T
        else:
            # subkernel = feat_temp@Message_all@feat_temp.T+ feat_temp@feat_temp.T            
            U1,S1,_ = np.linalg.svd(Message_all)
            H= U1@np.sqrt(np.diag(S1))@U1.T
            # H = 
            
            alpha = 1
            
            feat_temp_map = feat_temp@(alpha*H+np.eye(dim))
            # feat_temp_map = feat_temp@(alpha*H)
            # subkernel = feat_temp@Message_all@feat_temp.T+ feat_temp@feat_temp.T
            subkernel =feat_temp_map@feat_temp_map.T
        # subkernel = feat_temp@Message_all@feat_temp.T + feat_temp@feat_temp.T
        t = time.time()
        
        index_select = dpp(subkernel,num_candidate_temp)
        index_select = np.array(ave_index)[index_select]
        t3 = time.time()-t
        for dup_indx in select_cluster:
            index_select=np.delete(index_select,np.where(index_select==dup_indx))
        # print(len(index_select))
        # index_candidate[i,:len(index_select)]=index_select
        index_candidate[i,:len(index_select)]=index_select
        
        
        index_candidate[i,:len(index_select)]=index_select
        
        
        
        len_all = int( select_length)
        
        
        #calcluate the score function and find the optimized weight
        if num_cluster==1:
            return index_candidate[i,:len_all],(t1,t2,t3)
            # pass
        else:
            if programming==True:
                # print('here')
                for j in range(len(index_select)):
                    index_now = index_select[:j+1]
                    # cov_now = (feat_all_temp[index_now,:]@feat_all_temp[index_now,:].T)/(j+1)
                    # cov_now = (feat_all_temp[index_now,:]@feat_all_temp[index_now,:].T)
                    _,s_now,_ = np.linalg.svd(feat_all_temp[index_now,:])
                    # function_score[i,j] =(sum(np.log(epsilon*s_now+1/len_all))+ (dim-j-1)* np.log(1/len_all))*(j+1)/len_all
                    # function_score[i,j] =sum(np.log(epsilon*s_now+1))*(j+1)/len_all
                    function_score[i,j] =sum(np.log(epsilon*s_now**2+1))
                
                
    if programming==True:
        # find the weight
        Z_size = [len_all for _ in range(num_cluster)]
        Z = np.zeros(Z_size)
        
        for index, value in np.ndenumerate(Z):
            if sum(index)+num_cluster== len_all:
                for K in range(num_cluster):
                    Z[index] = Z[index]+ function_score[i,index[K]]     #

        opt = np.array(np.where(Z==np.nanmax(Z)))
        opt_num = opt.shape[1]
        
        Weight = opt[:,random.sample(range(opt_num),1)].reshape(-1)+np.ones(num_cluster)
                
                
            
    else:
        Weight=np.ones(num_cluster)
            
            
            
    if random_weight== True:
        Weight = np.random.rand(num_cluster)
        
    Weight =  Weight/sum(Weight)
    print(Weight)
    
    len_avai = len_all
   
    uniform_index = []

    for i in range(num_cluster):
        
        if i!= num_cluster-1:
            len_a = int(len_all*Weight[i])
            len_avai = len_avai-len_a
        else:
            len_a = len_avai
            
        if len_a>0:
            uniform_index.extend( list(index_candidate[i,:len_a]))
    
    uniform_index = np.array(uniform_index)
    # feat  = feat_all_temp[uniform_index,:]
    # subkernel =  feat@feat.T
    
    # uniform_index_order,dd = dpp_d(subkernel,max_length)
    # uniform_index_order = uniform_index[uniform_index_order]
    
    return uniform_index,(t1,t2,t3) #uniform_index_order
    
def MIMO_rand(select_index_set,feat_all,num_sample,select_length,num_candidate=100,sparse=1,svd_dim=0,programming=False, feedback = True): 

    
    dim = feat_all.shape[1]
    dim_dpp= int(np.sqrt((sparse-svd_dim)*dim*2))

    res_pxiel =  sparse*dim-  dim_dpp**2-svd_dim*dim
    
    
    
    # dim_dpp= 100

    # res_pxiel = 0
    
    epsilon = 1
    
    # net_glob = net_glob.to(device)
    # net_glob.eval()
    
    num_cluster = len(feat_all)//num_sample
    
    feat_all_temp = copy.deepcopy(feat_all)
    index_all = []


    index_candidate = -np.ones((num_cluster,num_candidate),dtype=int)*2
    
    
    function_score = -np.ones((num_cluster,num_candidate),dtype=float)
    function_score2 = -np.ones((num_cluster,num_candidate),dtype=float)
    
    d_all = []# np.zeros((num_cluster,num_candidate))
    
    index_all_clsuter = -np.ones((num_cluster,num_candidate),dtype=int)   #record slected index for each ClUSTER
    
    # index_all_clsuter = [[]*num_cluster]

    index_all_candidate = []

    density = np.zeros((num_cluster))


    select_cluster_index = np.array(select_index_set)//num_sample
    
    
    if select_index_set==[]:
        feedback = False
        
    if feedback == False:
        message = np.eye(dim)
        Message_all = message
        t1=0
        t2=0
    else:
        message_index = select_index_set
                        
        feat_all_temp = copy.deepcopy(feat_all)
        
        t = time.time()
        pre_code = np.eye(len(feat_all_temp[message_index,:].T))-feat_all_temp[message_index,:].T@np.linalg.inv(feat_all_temp[message_index,:]@feat_all_temp[message_index,:].T)@feat_all_temp[message_index,:]
        t1 = time.time()-t
        # active_dim = dpp(pre_code,dim_dpp)
    
        # if len(active_dim)<dim_dpp:
        #     print('error')
        #     return
        
        message = np.zeros((dim,dim))
        
        active_dim1 = random.sample(range(dim),dim_dpp)
        # active_dim2 = random.sample(range(dim),dim_dpp)
        # active_dim1  = np.random.choice(range(dim),dim_dpp**2*2)
        # active_dim2 = np.random.choice(range(dim),dim_dpp**2*2)
        active_dim2 = active_dim1
        
        
        
        count_a =0
        if dim_dpp>0:
            for m1 in range(dim):
                for m2 in range(dim):
                    if (m1 in active_dim1) and (m2 in active_dim2):
                        message[m1,m2] = pre_code[m1,m2]
                        message[m2,m1] = pre_code[m2,m1]
                        count_a = count_a+1
            
            
            # for ai in range(dim):
            #     m1 =active_dim1[ai]
            #     m2 =active_dim2[ai]
            #     if abs(message[m1,m2]-0)<1e-10:
            #         message[m1,m2] = pre_code[m1,m2]
            #         count_a = count_a+1
                
                if count_a==dim_dpp**2:
                    break
            # while count_a<dim_dpp**2:
            #     m1 = random.sample(range(dim),dim_dpp**2)
            #     m2 = random.sample(range(dim),dim_dpp**2)
            #     if abs(message[m1,m2]-0)>1e-10:
            #         count_a = count_a+1
            #         message[m1,m2] = pre_code[m1,m2]
            
        print(np.linalg.matrix_rank(message) )
        # print(f'count_a{count_a}')
        
        
        
        count_b =0
        
        while count_b<res_pxiel:
            m1 = random.sample(range(dim),1)
            m2 = random.sample(range(dim),1)
            if (m1 not in active_dim1) and (m2 not in active_dim2) and abs(message[m1,m2]-0)<1e-10:
                count_b= count_b+1
                message[m1,m2] = pre_code[m1,m2]
                
        # print(f'count{count_a+count_b}')
        # 
        
        precode_rest = pre_code-message
        Uma,Sma,_ = np.linalg.svd(precode_rest)
        
        
        message2 = Uma[:,:svd_dim]@np.diag(Sma[:svd_dim])@Uma[:,:svd_dim].T
        
        if svd_dim!=0:
            Message_all = message2+message
        
        else:
            Message_all = message
        t2 = time.time()-t
   #generate candidate
    for i in range(num_cluster):
        
        t = time.time()
        
        select_cluster_index_i = np.where(select_cluster_index==i)
        cluster_index = np.array((range(i*num_sample,i*num_sample+num_sample)))
        select_cluster = set(np.array(select_index_set)[select_cluster_index_i])
        # ave_index = set(cluster_index) - set(np.array(select_index_set)[select_cluster_index_i])
        ave_index = set(cluster_index)
        ave_index = np.array(list(ave_index))
        
        num_candidate_temp = min(num_candidate,len(ave_index))
    
        feat_temp = copy.deepcopy(feat_all_temp)[ave_index,:]
                    
        subkernel = feat_temp@Message_all@feat_temp.T+ feat_temp@feat_temp.T
        
        t = time.time()
        index_select = dpp(subkernel,num_candidate_temp)
        index_select = np.array(ave_index)[index_select]
        t3 = time.time()-t
        for dup_indx in select_cluster:
            index_select=np.delete(index_select,np.where(index_select==dup_indx))
        # print(len(index_select))
        # index_candidate[i,:len(index_select)]=index_select
        index_candidate[i,:len(index_select)]=index_select
        
        
        
        len_all = int( select_length)
        
        
        #calcluate the score function and find the optimized weight
        if num_cluster==1:
            return index_candidate[i,:len_all],(t1,t2,t3)
            # pass
        else:
            if programming==True:
                for j in range(len(index_select)):
                    index_now = index_select[:j+1]
                    # cov_now = (feat_all_temp[index_now,:]@feat_all_temp[index_now,:].T)/(j+1)
                    # cov_now = (feat_all_temp[index_now,:]@feat_all_temp[index_now,:].T)
                    _,s_now,_ = np.linalg.svd(feat_all_temp[index_now,:])
                    # function_score[i,j] =(sum(np.log(epsilon*s_now+1/len_all))+ (dim-j-1)* np.log(1/len_all))*(j+1)/len_all
                    # function_score[i,j] =sum(np.log(epsilon*s_now+1))*(j+1)/len_all
                    function_score[i,j] =sum(np.log(epsilon*s_now**2+1))
        
        
                
    if programming==True:
        # find the weight
        Z_size = [len_all for _ in range(num_cluster)]
        Z = np.zeros(Z_size)
        
        for index, value in np.ndenumerate(Z):
            if sum(index)+num_cluster== len_all:
                for K in range(num_cluster):
                    Z[index] = Z[index]+ function_score[i,index[K]]     #

        opt = np.array(np.where(Z==np.nanmax(Z)))
        opt_num = opt.shape[1]
        
        Weight = opt[:,random.sample(range(opt_num),1)].reshape(-1)+np.ones(num_cluster)
                
                
            
    else:
        Weight=np.ones(num_cluster)
            
            
            
        
    Weight =  Weight/sum(Weight)
    print(Weight)
    
    len_avai = len_all
   
    uniform_index = []

    for i in range(num_cluster):
        
        if i!= num_cluster-1:
            len_a = int(len_all*Weight[i])
            len_avai = len_avai-len_a
        else:
            len_a = len_avai
            
        if len_a>0:
            uniform_index.extend( list(index_candidate[i,:len_a]))
    
    uniform_index = np.array(uniform_index)
    # feat  = feat_all_temp[uniform_index,:]
    # subkernel =  feat@feat.T
    
    # uniform_index_order,dd = dpp_d(subkernel,max_length)
    # uniform_index_order = uniform_index[uniform_index_order]
    
    return uniform_index,(t1,t2,t3) #uniform_index_order
    
   
    
   
    
   
    
   
    