# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:33:48 2023

@author: xiwenc
"""

import numpy as np
import copy
import math,time



from torchvision import datasets, transforms
import torch


import torch.nn.functional as F













# XX = np.random.rand(100,1000)
# kernel_matrix = XX@XX.T

# dpp_min(kernel_matrix, 3, epsilon=1E-20)


def dpp_min(kernel_matrix, max_length, epsilon=1E-20):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix)) #shape(item_size,)
    # print(di2s)
    selected_items = list()
    selected_item = np.argmin(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        # print(len(di2s))
        selected_item = np.argsort(di2s)[len(selected_items)]
        # print(np.sort(di2s))
        # print(di2s[selected_item])
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items















def dpp_I(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    # kernel_matrix = copy.deepcopy(kernel_matrix)+np.eye(len(kernel_matrix))
    
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))+1 #shape(item_size,)
    # print(di2s)
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        # print(len(di2s))
        selected_item = np.argmax(di2s)
        # print(di2s[selected_item])
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix)) #shape(item_size,)
    # print(di2s)
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        # print(len(di2s))
        selected_item = np.argmax(di2s)
        # print(di2s[selected_item])
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def dpp_d(kernel_matrix, max_length, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    d_list = []
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix)) #shape(item_size,)
    # print(di2s)
    selected_items = list()
    selected_item = np.argmax(di2s)
    d_list.append(di2s[selected_item])
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        # print(len(di2s))
        selected_item = np.argmax(di2s)
        # print(di2s[selected_item])
        d_list.append(di2s[selected_item])
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items,d_list


def div(feat_all,index_out):
       select = np.take(feat_all,index_out,axis=(0))
       u,s,_ = np.linalg.svd(select@select.T)
       div_emsimate =sum(np.log(s))
       return div_emsimate
   
    
def gain(now_select,privous_select):
    # now_select = feat_all_temp[index_select,:]
    # privous_select = feat_all[index_all,:]
    X = np.concatenate((now_select,privous_select), axis=0)
    _,s,_ = np.linalg.svd(np.eye(X.shape[1])+ X.shape[1]/len(X)*X.T@X)
    R = sum(np.log(s))

    return R

def rate(X):
    # now_select = feat_all_temp[index_select,:]
    # privous_select = feat_all[index_all,:]
    # X = np.concatenate((now_select,privous_select), axis=0)
    _,s,_ = np.linalg.svd(np.eye(X.shape[1])+ X.shape[1]/len(X)*X.T@X)
    R = sum(np.log(s))

    return R


def softmax(x):
    return(np.exp(x)/(np.exp(x).sum()+1e-8))





        
        
        
        