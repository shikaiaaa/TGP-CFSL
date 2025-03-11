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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# torch.backends.cudnn.enabled = False

def initialize(src_dim, tar_dim, class_n, test_class_n, learning_rate):
    utils.same_seeds(0)
    def _init_():
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('classificationMap'):
            os.makedirs('classificationMap')
    _init_()

    parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
    parser.add_argument("-f", "--feature_dim", type=int, default=160)  # 特征维度
    parser.add_argument("-c", "--src_input_dim", type=int, default=src_dim)  # Chikusei data set 的 128个 spectral bands
    parser.add_argument("-d", "--tar_input_dim", type=int, default=tar_dim)  # PaviaU=103 spectral bands
    parser.add_argument("-n", "--n_dim", type=int, default=100)  # transformed data set 的 spectral bands
    parser.add_argument("-w", "--class_num", type=int, default=class_n)  # 从源域数据集19个类中选择9类为一个9-way 1-shot 的 task
    parser.add_argument("-s", "--shot_num_per_class", type=int, default=1)  # 每一个task中，support set 包含1个标签样本
    parser.add_argument("-b", "--query_num_per_class", type=int, default=19)  # 每一个task中，query set 包含19个标签样本
    parser.add_argument("-e", "--episode", type=int, default=10000)  # 共进行30000次训练task
    parser.add_argument("-l", "--learning_rate", type=float, default=learning_rate)  # 学习率
    # target
    parser.add_argument("-m", "--test_class_num", type=int, default=test_class_n)  # 目标域PaviaU中的类别个数
    parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help='5 4 3 2 1')  # 测试时目标域中选取有标签样本support set的个数
    args = parser.parse_args(args=[])

    return args
# get source domain data and test domain data
def load_sourcedata(source_data_path, source_graph_path, targetdata_path, targetlabel_path):
    # load source domain data set
    with open(source_data_path, 'rb') as handle:
        source_imdb = pickle.load(handle)
    with open(source_graph_path, 'rb') as handle:
        source_graph = pickle.load(handle)

    # process source domain data set
    data_train = source_imdb['data'] # (77592, 9, 9, 128)
    labels_train = source_imdb['Labels'] # 77592
    graph_train = source_graph['graph']

    keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
    label_encoder_train = {}
    for i in range(len(keys_all_train)):
        label_encoder_train[keys_all_train[i]] = i

    train_set = {}
    Graph_set = {}
    for class_, path, graph in zip(labels_train, data_train, graph_train):
        if label_encoder_train[class_] not in train_set:
            train_set[label_encoder_train[class_]] = []
            Graph_set[label_encoder_train[class_]] = []
        train_set[label_encoder_train[class_]].append(path)
        Graph_set[label_encoder_train[class_]].append(graph)

    data = train_set
    graph = Graph_set
    del Graph_set
    del train_set
    del keys_all_train
    del label_encoder_train

    data = utils.sanity_check(data) # 200 labels samples per class
    graph = utils.sanity_check(graph)

    for class_ in data:
        for i in range(len(data[class_])):
            image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
            data[class_][i] = image_transpose
            graph_transpose = np.transpose(graph[class_][i], (1, 0))
            graph[class_][i] = graph_transpose

    Data_Band_Scaler, GroundTruth = utils.load_data(targetdata_path, targetlabel_path)
    return data, graph, Data_Band_Scaler, GroundTruth
# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, iDataSet, train_test_path):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    HalfWidth = 4

    [Row, Column] = np.nonzero(GroundTruth)  # (10249,) (10249,)

    nSample = np.size(Row)
    print('number of sample', nSample)

    m = int(np.max(GroundTruth))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    a = None
    with open(train_test_path+str(iDataSet), 'rb') as handle:
        a = pickle.load(handle)

    print('the number of train_indices:', len(a['train_data']))  # 520
    print('the number of test_indices:', len(a['test_data']))  # 9729
    print('the number of da_train_data after data argumentation:', len(a['da_train_data']))  # 520

    nTrain = len(a['train_data'])
    nTest = len(a['test_data'])
    da_nTrain = len(a['da_train_data'])

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['graph'] = np.zeros([nTrain + nTest, SuperPixelCount, nBand], dtype=np.int64)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)
    #
    RandPermData = 1
    for iSample in range(nTrain + nTest):
        if iSample < len(a['train_data']):
            imdb['data'][:, :, :, iSample] = a['train_data'][iSample]
            imdb['Labels'][iSample] = a['train_labels'][iSample]
            imdb['graph'][iSample, :, :, ] = a['train_graph'][iSample]
        else:
            i = iSample - len(a['train_data'])
            imdb['data'][:, :, :, iSample] = a['test_data'][i]
            imdb['Labels'][iSample] = a['test_labels'][i]
            imdb['graph'][iSample, :, :, ] = a['test_graph'][i]

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    #print('utils.matcifar.')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False)
    del train_dataset
    #print('del train_dataset.')

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    #print('utils.matcifar.')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    del test_dataset
    del imdb
    #print('del test_dataset.del imdb.')

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)
    graph_da_train = {}
    graph_da_train['graph'] = np.zeros([da_nTrain, SuperPixelCount, nBand], dtype=np.int64)

    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            a['da_train_data'][iSample]
        )
        imdb_da_train['Labels'][iSample] = a['da_train_labels'][iSample]
        graph_da_train['graph'][iSample] = a['da_train_graph'][iSample]

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1 # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    del a
    print('ok')

    return train_loader, test_loader, imdb_da_train, graph_da_train, RandPermData, Row, Column, ++nTrain
def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, iDataSet, train_test_path):
    train_loader, test_loader, imdb_da_train, graph_da_train, RandPerm,Row, Column,nTrain =\
        get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth,
         class_num=class_num,shot_num_per_class=shot_num_per_class, iDataSet=iDataSet, train_test_path=train_test_path)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels, train_graph = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])
    print('size of train datas:', train_graph.shape)

    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    target_da_graph = np.transpose(graph_da_train['graph'], (0, 2, 1))
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    target_da_Graph_set = {}
    for class_, path, graph in zip(target_da_labels, target_da_datas, target_da_graph):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
            target_da_Graph_set[class_] = []
        target_da_train_set[class_].append(path)
        target_da_Graph_set[class_].append(graph)
    target_da_metatrain_data = target_da_train_set
    target_da_metatrain_graph = target_da_Graph_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, target_da_metatrain_graph,RandPerm,Row, Column,nTrain
# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()
        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

        self.layer_last = nn.Sequential(nn.Linear(in_features=self._get_layer_size()[1],
                                                  out_features=128,
                                                  bias=True),
                                        nn.BatchNorm1d(128))

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1, 100, 9, 9))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            _, t, c, w, h = x.size()
            s1 = t * c * w * h
            x = self.conv(x)
            x = x.view(x.shape[0],-1)
            s2 = x.size()[1]
        return s1, s2

    def forward(self, x):  # x:(400,100,9,9)
        # x = x.unsqueeze(1)  # (400,1,100,9,9)
        x = self.block1(x)  # (1,8,100,9,9)

        x = self.maxpool1(x)  # (1,8,25,5,5)
        x = self.block2(x)  # (1,16,25,5,5)
        x = self.maxpool2(x)  # (1,16,7,3,3)
        x = self.conv(x)  # (1,32,5,1,1)
        x = x.view(x.shape[0], -1)  # (1,160)
        x = self.layer_last(x)
        return x
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x
class Graph_Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Graph_Mapping, self).__init__()
        self.preconv = nn.Conv1d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm1d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self, ):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.target_graph_mapping = Graph_Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.source_graph_mapping = Graph_Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.avgpool3d1 = nn.AdaptiveAvgPool3d((25, 5, 5))
        self.avgpool3d2 = nn.AdaptiveAvgPool3d((7, 3, 3))
        self.conv = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, bias=False)
        self.avgpool2d1 = nn.AdaptiveAvgPool2d((25, SuperPixelCount))
        self.avgpool2d2 = nn.AdaptiveAvgPool2d((7, SuperPixelCount))
        self.linear = nn.Linear(100, 4)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mixconvList = nn.Sequential(CGConvBlock(1, 8, 3),
                                         CGConvBlock(8, 16, 3),
                                         CGConvBlock(16, 32, 3),
                                         CGConvBlock(16, 32, 3),
                                         CGConvBlock(64, 128, 3))
        self.embedding=nn.Sequential(nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 128, kernel_size=3),
                                    nn.Tanh(),
                                    nn.AdaptiveAvgPool2d(1))

    def forward(self, data: torch.Tensor, subgraphs:torch.Tensor, domain='source'):  # x
        # print(x.shape)
        if domain == 'target':
            data = self.target_mapping(data)  # (45, 100,9,9)
            subgraphs = self.target_graph_mapping(subgraphs.type(torch.FloatTensor).cuda())

        elif domain == 'source':
            data = self.source_mapping(data)  # (45, 100,9,9)
            subgraphs = self.source_graph_mapping(subgraphs.type(torch.FloatTensor).cuda())
        #
        data = data.unsqueeze(1) #(9,100,9,9)→(9,1,100,9,9)
        subgraphs = subgraphs.unsqueeze(1)

        for i in range(3):
            data, subgraphs = self.mixconvList[i](data, subgraphs)

        data = self.pool(data).squeeze(-1).squeeze(-1)
        data = self.linear(data)
        data = data.view(data.shape[0], -1)
        return data

class DFSL(nn.Module):
    def __init__(self, ):
        super(DFSL, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, data: torch.Tensor, subgraphs:torch.Tensor, domain='source'):  # x

        if domain == 'target':
            data = self.target_mapping(data)  # (45, 100,9,9)

        elif domain == 'source':
            data = self.source_mapping(data)  # (45, 100,9,9)

        data = data.unsqueeze(1)
        data = self.feature_encoder(data)

        return data

class DomainClassifier(nn.Module):
    def __init__(self, input_nc=128, output_nc=128):
        super(DomainClassifier, self).__init__()
        self.liner1 = nn.Linear(in_features=input_nc, out_features=input_nc // 2)
        self.liner2 = nn.Linear(in_features=input_nc // 2, out_features=output_nc)

    def forward(self, x):
        x_ci = F.relu(self.liner1(x), inplace=True)
        x_ci = F.relu(self.liner2(x_ci), inplace=True)
        return x_ci
class Metric_encoder(nn.Module):
    """docstring for RelationNetwork为每个类学习一个类注意力权重"""
    def __init__(self):
        super(Metric_encoder, self).__init__()
        self.inchannel = 160 + CLASS_NUM
        self.f_psi = nn.Sequential(
            nn.Linear(self.inchannel, 160),
            nn.BatchNorm1d(160),
            nn.Sigmoid()
        )#MLP
    def forward(self, s, q, sl):#q(304,160) s(16,160) sl(16,)
        sl = torch.as_tensor(sl, dtype=torch.long).cuda()
        sl = F.one_hot(sl) #(16,16)
        ind = torch.cat((s, sl), 1) #(16,176)
        weight_ = self.f_psi(ind) #(16,160)
        attention_ = weight_.unsqueeze(0).repeat(q.shape[0], 1, 1)
        match_ = euclidean_metric(q, s)  # (304,16,160)
        attention_match_score = torch.mul(attention_, match_)  # (304,16,160)
        score = torch.sum(attention_match_score.contiguous(), dim=2)  # (304,16)
        return score
def euclidean_metric(a, b):

    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    # logits = -((a - b) ** 2)
    return logits
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

SuperPixelCount = 25
# for (FLAG, nDataSet) in [(1, 10)]:
for (FLAG, nDataSet) in [(4, 5)]:
    if FLAG == 1:#1-UP;
        # load data
        source_data_path = os.path.join('datasets', 'Chikusei_imdb_128_sk_25.pickle')
        source_graph_path = os.path.join('datasets', 'Chikusei_Graphs_128_sk_25.pickle')
        targetdata_path = 'datasets/paviaU/paviaU.mat'
        targetlabel_path = 'datasets/paviaU/paviaU_gt.mat'
        train_test_path = '../datasets/UP_'
        args = initialize(src_dim=128, tar_dim=103, class_n=9, test_class_n=9, learning_rate=0.0005)
        metatrain_data, metatrain_graph, Data_Band_Scaler, GroundTruth = load_sourcedata(source_data_path, source_graph_path, targetdata_path, targetlabel_path)
        PATH1 = 'checkpoints/DGIP_3DCNN_UP_OURS_feature_encoder'
        pass
    if FLAG == 2:#2-IP; #69.64%
        # load data
        source_data_path = os.path.join('datasets', 'Chikusei_imdb_128_sk_25.pickle')
        source_graph_path = os.path.join('datasets', 'Chikusei_Graphs_128_sk_25.pickle')
        targetdata_path = 'datasets/IP/indian_pines_corrected.mat'
        targetlabel_path = 'datasets/IP/indian_pines_gt.mat'
        train_test_path = '../datasets/IP_'
        args = initialize(src_dim=128, tar_dim=200, class_n=16, test_class_n=16, learning_rate=0.0005)
        metatrain_data, metatrain_graph, Data_Band_Scaler, GroundTruth = load_sourcedata(source_data_path, source_graph_path, targetdata_path, targetlabel_path)
        PATH1 = 'checkpoints/DGIP_3DCNN_IP_OURS_feature_encoder'
        pass
    if FLAG == 3:#3-salinas
        # load data
        source_data_path = os.path.join('datasets', 'Chikusei_imdb_128_sk_25.pickle')
        source_graph_path = os.path.join('datasets', 'Chikusei_Graphs_128_sk_25.pickle')
        targetdata_path = 'datasets/salinas/salinas_corrected.mat'
        targetlabel_path = 'datasets/salinas/salinas_gt.mat'
        train_test_path = '../datasets/Salinas_'
        args = initialize(src_dim=128, tar_dim=204, class_n=16, test_class_n=16, learning_rate=0.0005)
        metatrain_data, metatrain_graph, Data_Band_Scaler, GroundTruth = load_sourcedata(source_data_path, source_graph_path, targetdata_path, targetlabel_path)
        PATH1 = 'checkpoints/DGIP_3DCNN_Salinas_IP_OURS_feature_encoder'
        pass
    if FLAG == 4:#4-PC
        # load data
        source_data_path = os.path.join('datasets', 'Chikusei_imdb_128_sk_25.pickle')
        source_graph_path = os.path.join('datasets', 'Chikusei_Graphs_128_sk_25.pickle')
        targetdata_path = 'datasets/pavia/pavia.mat'
        targetlabel_path = 'datasets/pavia/pavia_gt.mat'
        train_test_path = 'datasets/PC_'
        args = initialize(src_dim=128, tar_dim=102, class_n=9, test_class_n=9, learning_rate=0.0005)
        metatrain_data, metatrain_graph, Data_Band_Scaler, GroundTruth = load_sourcedata(source_data_path, source_graph_path, targetdata_path, targetlabel_path)
        PATH1 = 'checkpoints/Final_PC_OURS_feature_encoder.pkl'
        PATH2 = 'checkpoints/Final_PC_OURS_domain_classifier.pkl'
        PATH3 = 'checkpoints/Final_PC_OURS_metric_net_encoder.pkl'
        pass

# Hyper Parameters
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1
crossEntropy = nn.CrossEntropyLoss().cuda()
criterion = torch.nn.MSELoss().cuda()
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

# first 1330
seeds = [1532, 1330, 6235, 1336, 1535, 1224, 1236, 1226, 1345, 1233]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    #np.random.seed(seeds[4])
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_da_metatrain_graph, RandPerm,Row, Column, nTrain = \
        get_target_dataset(Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,
        class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS, iDataSet=iDataSet, train_test_path=train_test_path)
    # model
    feature_encoder = Network()
    # feature_encoder = nn.DataParallel(feature_encoder )
    feature_encoder.cuda()
    feature_encoder.train()

    Block_src = DualGraph(1, 0.1, CLASS_NUM * SHOT_NUM_PER_CLASS,
                          CLASS_NUM * SHOT_NUM_PER_CLASS + CLASS_NUM * QUERY_NUM_PER_CLASS,).cuda()

    Block_tar = DualGraph(1, 0.1, CLASS_NUM * SHOT_NUM_PER_CLASS,
                          CLASS_NUM * SHOT_NUM_PER_CLASS + CLASS_NUM * QUERY_NUM_PER_CLASS,).cuda()
    AttentionW = AttentionalGNN(CLASS_NUM, 128, ['cross'] * 1).cuda()

    Block_src.train()
    Block_tar.train()
    AttentionW.train()

    Block_src_optim = torch.optim.Adam(Block_src.parameters(), lr=0.01, weight_decay=1e-4)
    Block_tar_optim = torch.optim.Adam(Block_tar.parameters(), lr=0.01, weight_decay=1e-4)
    AttentionW_optim = torch.optim.Adam(AttentionW.parameters(), lr=0.01, weight_decay=1e-4)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate, weight_decay=0.1)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []
    train_start = time.time()

    for episode in range(EPISODE):
        # source domain few-shot + domain adaptation
        if episode % 1 == 0:
            '''Few-shot claification'''
            # get few-shot classification samples
            task1 = utils.Task(metatrain_data, metatrain_graph, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader1 = utils.get_HBKC_data_loader(task1, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader1 = utils.get_HBKC_data_loader(task1, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
            # sample datas
            supports1, support_labels1, support_graphs1 = support_dataloader1.__iter__().next()  # (5, 100, 9, 9)
            querys1, query_labels1, query_graphs1 = query_dataloader1.__iter__().next()  # (75,100,9,9)
            # calculate features
            support_features1 = feature_encoder(supports1.cuda(), support_graphs1.cuda(), domain='source')  # torch.Size([409, 32, 7, 3, 3])
            query_features1 = feature_encoder(querys1.cuda(), query_graphs1.cuda(), domain='source')  # torch.Size([409, 32, 7, 3, 3])

            task2 = utils.Task(target_da_metatrain_data, target_da_metatrain_graph, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader2 = utils.get_HBKC_data_loader(task2, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader2 = utils.get_HBKC_data_loader(task2, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
            # sample datas
            supports2, support_labels2, support_graphs2 = support_dataloader2.__iter__().next()  # (5, 100, 9, 9)
            querys2, query_labels2, query_graphs2 = query_dataloader2.__iter__().next()  # (75,100,9,9)
            # calculate features
            support_features2 = feature_encoder(supports2.cuda(), support_graphs2.cuda(), domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features2 = feature_encoder(querys2.cuda(), query_graphs2.cuda(), domain='target')

            # initialize nodes and edges for dual graph model
            tensors_src = utils.allocate_tensors()
            for key, tensor in tensors_src.items():
                tensors_src[key] = tensor.cuda()
            batch_src = [support_features1.unsqueeze(0).unsqueeze(0),
                         support_labels1.unsqueeze(0).unsqueeze(0), query_features1.unsqueeze(0).unsqueeze(0),
                         query_labels1.unsqueeze(0).unsqueeze(0)]
            batch_tar = [support_features2.unsqueeze(0).unsqueeze(0),
                         support_labels2.unsqueeze(0).unsqueeze(0), query_features2.unsqueeze(0).unsqueeze(0),
                         query_labels2.unsqueeze(0).unsqueeze(0)]
            support_data_src, support_label_src, query_data_src, query_label_src, all_data_src, all_label_in_edge_src, node_feature_gd_src, \
            edge_feature_gp_src, edge_feature_gd_src = utils.initialize_nodes_edges(batch_src,
                                                                                    CLASS_NUM,
                                                                                    tensors_src,
                                                                                    1,
                                                                                    query_labels1.shape[0],
                                                                                    CLASS_NUM,
                                                                                    device)
            tensors_tar = utils.allocate_tensors()
            for key, tensor in tensors_tar.items():
                tensors_tar[key] = tensor.cuda()
            support_data_tar, support_label_tar, query_data_tar, query_label_tar, all_data_tar, all_label_in_edge_tar, node_feature_gd_tar, \
            edge_feature_gp_tar, edge_feature_gd_tar = utils.initialize_nodes_edges(batch_tar,
                                                                                    CLASS_NUM,
                                                                                    tensors_tar,
                                                                                    1,
                                                                                    query_labels2.shape[0],
                                                                                    CLASS_NUM,
                                                                                    device)

            # calculate prototype
            support_proto_src = support_features1
            support_proto_tar = support_features2

            # the IDE-block
            last_layer_data_src = torch.cat((support_features1, query_features1)).unsqueeze(0)

            last_layer_data_tar = torch.cat((support_features2, query_features2)).unsqueeze(0)

            _, _, _, point_nodes_src, distribution_nodes_src = Block_src(last_layer_data_src,
                                                                         last_layer_data_src,
                                                                         node_feature_gd_src,
                                                                         edge_feature_gd_src,
                                                                         edge_feature_gp_src)

            _, _, _, point_nodes_tar, distribution_nodes_tar = Block_tar(last_layer_data_tar,
                                                                         last_layer_data_tar,
                                                                         node_feature_gd_tar,
                                                                         edge_feature_gd_tar,
                                                                         edge_feature_gp_tar)

            cross_att_loss = AttentionW(point_nodes_src, point_nodes_tar, distribution_nodes_src,
                                        distribution_nodes_tar)

            logits1 = euclidean_metric(query_features1, support_features1)
            f_loss1 = crossEntropy(logits1, query_labels1.long().cuda())

            logits2 = euclidean_metric(query_features2, support_features2)
            f_loss2 = crossEntropy(logits2, query_labels2.long().cuda())

            f_loss = f_loss1 + f_loss2 + cross_att_loss
            # f_loss = f_loss1 + f_loss2

            feature_encoder.zero_grad()
            Block_src.zero_grad()
            Block_tar.zero_grad()
            AttentionW.zero_grad()
            f_loss.backward()
            feature_encoder_optim.step()
            Block_src_optim.step()
            Block_tar_optim.step()
            AttentionW_optim.step()

            logits = torch.cat((logits1, logits2),0)
            query_labels = torch.cat((query_labels1, query_labels2),0)
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += query_labels.shape[0]

        if (episode + 1) % 1000 == 0 or episode == 0:  # display`
            train_loss.append(f_loss.item())
            print('episode {:>3d}: cross_att_loss: {:6.4f}, f_loss: {:6.4f}, acc {:6.4f}'.format(episode + 1, cross_att_loss.item(), f_loss.item(), total_hit / total_num))

        if (episode + 1) % 1000 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels, train_graph = train_loader.__iter__().next()
            train_features = feature_encoder(Variable(train_datas).cuda(), Variable(train_graph).cuda(), domain='target')  # (45, 160)
            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            temp = train_features.cpu().detach().numpy()
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()

            test_start = time.time()

            for test_datas, test_labels, test_graph in test_loader:
                batch_size = test_labels.shape[0]

                test_features = feature_encoder(Variable(test_datas).cuda(), Variable(test_graph).cuda(), domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            temp = len(test_loader.dataset)
            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset), 100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), str(PATH1+str(iDataSet)+".pkl"))
                print("save networks for episode:",episode+1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_RandPerm,best_Row, best_Column,best_nTrain = RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))

    del feature_encoder
    del Block_src
    del Block_tar
    del AttentionW
    torch.cuda.empty_cache()

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("finetuning time per DataSet(s): " + "{:.5f}".format(test_start-train_end))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-test_start))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

