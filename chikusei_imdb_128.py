import numpy as np
from sklearn.decomposition import PCA
import random
import pickle
import h5py
import hdf5storage
from sklearn import preprocessing
import scipy.io as sio
from SubGraphSampler import SubGraphSampler
from utils import *
import concurrent.futures

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(groundTruth):
    labels_loc = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

    whole_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]

    np.random.shuffle(whole_indices)
    return whole_indices


def load_data_HDF(image_file, label_file):
    image_data = hdf5storage.loadmat(image_file)
    label_data = hdf5storage.loadmat(label_file)
    data_all = image_data['chikusei']  # data_all:ndarray(2517,2335,128)
    label = label_data['GT'][0][0][0]  # label:(2517,2335)

    [nRow, nColumn, nBand] = data_all.shape
    print('chikusei', nRow, nColumn, nBand)

    data_all = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    print(data_all.shape)
    data_scaler = preprocessing.scale(data_all)
    data_scaler = data_scaler.reshape(2517, 2335, 128)

    return data_scaler, label

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]
    label = label_data[label_key]
    gt = label.reshape(np.prod(label.shape[:2]), )

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    print(data.shape)
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return data_scaler, gt

def getDataAndLabels(trainfn1, trainfn2):
    if ('Chikusei' in trainfn1 and 'Chikusei' in trainfn2):
        Data_Band_Scaler, gt = load_data_HDF(trainfn1, trainfn2)
    else:
        Data_Band_Scaler, gt = load_data(trainfn1, trainfn2)

    del trainfn1, trainfn2
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    GD = GetDataset(Data_Band_Scaler, extend=4)
    trainDatasetList = GD.getAllSamples(gt)

    SS = SubGraphSampler(Data_Band_Scaler)
    SS.ConstructGraph(knn_neighbors=25, n_avgsize=64, show_segmap=True)
    trainGraphsList = SS.get_Allgraph_samples(gt=gt, num_hops=1)

    print('selectNeighboringPatch is ok')
    nSample = len(trainDatasetList)
    imdb = {}
    imdb['data'] = np.zeros([nSample, 2 * 4 + 1, 2 * 4 + 1, nBand], dtype=np.float32)  # <class 'tuple'>: (9, 9, 100, 77592)
    imdb['Labels'] = np.zeros([nSample], dtype=np.int64)  # <class 'tuple'>: (77592,)
    imdb['set'] = np.zeros([nSample], dtype=np.int64)
    Graphs = {}
    Graphs['graph'] = np.zeros([nSample, 25, nBand], dtype=np.int64)

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :, ] = trainDatasetList[iSample][0]
        imdb['Labls'][iSample] = trainDatasetList[iSample][1]
        Graphs['graph'][iSample, :, :, ] = trainGraphsList[iSample]
        if iSample % 100 == 0:
            print('iSample', iSample)
    imdb['set'] = np.ones([nSample]).astype(np.int64)

    print('Data is OK.')

    return imdb, Graphs

train_data_file = 'datasets/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat'
train_label_file = 'datasets/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'

imdb, Graphs = getDataAndLabels(train_data_file, train_label_file)

with open('datasets/Chikusei_imdb_128_sk_25.pickle', 'wb') as handle:
    pickle.dump(imdb, handle, protocol=4)
with open('datasets/Chikusei_Graphs_128_sk_25.pickle', 'wb') as handle:
    pickle.dump(Graphs, handle, protocol=4)

print('Images preprocessed')