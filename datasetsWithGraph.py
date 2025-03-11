import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os
import math
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
from SubGraphSampler import SubGraphSampler
from utils import *
from Networks import *
from DGIP2 import DualGraph
from DGIP1 import AttentionalGNN

TEST_LSAMPLE_NUM_PER_CLASS = 5

def load_sourcedata(source_path, targetdata_path, targetlabel_path):
    # load source domain data set
    with open(source_path, 'rb') as handle:
        source_imdb = pickle.load(handle)

    # process source domain data set
    data_train = source_imdb['data'] # (77592, 9, 9, 128)
    labels_train = source_imdb['Labels'] # 77592
    keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
    label_encoder_train = {}
    for i in range(len(keys_all_train)):
        label_encoder_train[keys_all_train[i]] = i

    train_set = {}
    for class_, path in zip(labels_train, data_train):
        if label_encoder_train[class_] not in train_set:
            train_set[label_encoder_train[class_]] = []
        train_set[label_encoder_train[class_]].append(path)
    data = train_set
    del train_set
    del keys_all_train
    del label_encoder_train

    data = utils.sanity_check(data) # 200 labels samples per class

    for class_ in data:
        for i in range(len(data[class_])):
            image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
            data[class_][i] = image_transpose

    # source few-shot classification data
    metatrain_data = data
    del data

    Data_Band_Scaler, GroundTruth = utils.load_data(targetdata_path, targetlabel_path)
    return metatrain_data, Data_Band_Scaler, GroundTruth

def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, iDataSet, seeds):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    [Row, Column] = np.nonzero(GroundTruth)  # (10249,) (10249,)

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_trainDatasetList = {}
    da_trainGraphsList = {}
    trainDatasetList = {}
    trainGraphsList = {}
    testDatasetList = {}
    testGraphsList = {}
    m = int(np.max(GroundTruth))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    GD = GetDataset(Data_Band_Scaler, extend=4)
    SS = SubGraphSampler(Data_Band_Scaler)
    SS.ConstructGraph(knn_neighbors=25, n_avgsize=64, show_segmap=True)

    np.random.seed(seeds[iDataSet])

    for i in range(m):
        train_gt = np.zeros((nRow, nColumn))
        indices = [j for j in range(len(Row)) if GroundTruth[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        for j in train[i]:
            train_gt[Row[j], Column[j]] = GroundTruth[Row[j], Column[j]]
        trainDatasetList[i] = GD.getAllSamples(train_gt)
        trainGraphsList[i] = SS.get_Allgraph_samples(gt=train_gt, num_hops=1)
        da_trainDatasetList[i] = []
        da_trainGraphsList[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_trainDatasetList[i] += trainDatasetList[i]
            da_trainGraphsList[i] += trainGraphsList[i]
        test[i] = indices[nb_val:]
        train_gt = np.zeros((nRow, nColumn))
        for j in test[i]:
            train_gt[Row[j], Column[j]] = GroundTruth[Row[j], Column[j]]
        testDatasetList[i] = GD.getAllSamples(train_gt)
        testGraphsList[i] = SS.get_Allgraph_samples(gt=train_gt, num_hops=1)
        print(i)

    da_train_data = []
    da_train_graph = []
    train_data = []
    train_graph = []
    test_data = []
    test_graph = []
    for i in range(m):
        train_data += trainDatasetList[i]
        train_graph += trainGraphsList[i]
        test_data += testDatasetList[i]
        test_graph += testGraphsList[i]
        da_train_data += da_trainDatasetList[i]
        da_train_graph += da_trainGraphsList[i]

    del trainDatasetList
    del testDatasetList
    del da_trainDatasetList
    del trainGraphsList
    del testGraphsList
    del da_trainGraphsList

    nTrain = len(train_data)
    nTest = len(test_data)
    da_nTrain = len(da_train_data)
    imdb = {}
    imdb['train_data'] = np.zeros([nTrain, 9, 9, nBand], dtype=np.float32)
    imdb['train_graph'] = np.zeros([nTrain, 25, nBand], dtype=np.int64)
    imdb['train_labels'] = np.zeros([nTrain], dtype=np.int64)
    imdb['test_data'] = np.zeros([nTest, 9, 9, nBand], dtype=np.float32)
    imdb['test_graph'] = np.zeros([nTest, 25, nBand], dtype=np.int64)
    imdb['test_labels'] = np.zeros([nTest], dtype=np.int64)
    imdb['da_train_data'] = np.zeros([da_nTrain, 9, 9, nBand], dtype=np.float32)
    imdb['da_train_graph'] = np.zeros([da_nTrain, 25, nBand], dtype=np.int64)
    imdb['da_train_labels'] = np.zeros([da_nTrain], dtype=np.int64)

    for i in range(nTrain):
        imdb['train_data'][i, :, :, :, ] = train_data[i][0]
        imdb['train_graph'][i, :, :, ] = train_graph[i][0]
        imdb['train_labels'][i] = train_data[i][1]
    for i in range(nTest):
        imdb['test_data'][i, :, :, :, ] = test_data[i][0]
        imdb['test_graph'][i, :, :, ] = test_graph[i][0]
        imdb['test_labels'][i] = test_data[i][1]
    for i in range(da_nTrain):
        imdb['da_train_data'][i, :, :, :, ] = da_train_data[i][0]
        imdb['da_train_graph'][i, :, :, ] = da_train_graph[i][0]
        imdb['da_train_labels'][i] = da_train_data[i][1]

    del train_data
    del train_graph
    del test_data
    del test_graph
    del da_train_data
    del da_train_graph
    gc.collect()
    print("data is ok")
    with open('../datasets/PC_' + str(iDataSet), 'wb') as handle:
        pickle.dump(imdb, handle, protocol=4)
    # np.random.shuffle(test_indices)

if __name__ == '__main__':
    source_path = os.path.join('datasets', 'Chikusei_imdb_128_6.pickle')
    targetdata_path = 'datasets/pavia/pavia.mat'
    targetlabel_path = 'datasets/pavia/pavia_gt.mat'
    _, Data_Band_Scaler, GroundTruth = load_sourcedata(source_path, targetdata_path, targetlabel_path)
    class_num = 9
    shot_num_per_class = 5
    seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
    for iDataSet in range(10):
        get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, iDataSet, seeds)
        print("iDataSet:" + str(iDataSet))