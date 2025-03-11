import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.utils.data as data
import cv2
import spectral as spy
from OT_torch_ import  cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform

# 设置随机种子
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

class classification(data.Dataset):
    def __init__(self, imdb):
        self.data = imdb['data']
        self.labels = imdb['labels']
        self.graph = imdb['graph']
        self.graph = self.graph.transpose((0, 2, 1))
        self.data = self.data.transpose((3, 2, 0, 1))
    def __getitem__(self, index):
        img, label, graph = self.data[index], self.labels[index], self.graph[index]
        return img, label, graph

    def __len__(self):
        return len(self.data)

class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.train_graph = self.imdb['graph'][self.x1, :, :]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]
            self.test_graph = self.imdb['graph'][self.x2, :, :]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.train_graph = self.imdb['graph'][self.x1, :, :]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            self.test_graph = self.imdb['graph'][self.x2, :, :]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)
                self.train_graph = self.train_graph.transpose((0, 2, 1))
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
                self.test_graph = self.test_graph.transpose((0, 2, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.train_graph = self.train_graph.transpose((0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))
                self.test_graph = self.test_graph.transpose((0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, graph = self.train_data[index], self.train_labels[index], self.train_graph[index]
        else:
            img, target, graph = self.test_data[index], self.test_labels[index], self.test_graph[index]
        return img, target, graph

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

class Task(object):

    def __init__(self, data, graph, num_classes, shot_num, query_num):
        self.data = data
        self.graph = graph
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        self.support_graphs = []
        self.query_graphs = []
        for c in class_list:
            temp = self.data[c]  # list
            temp2 = self.graph[c]

            self.support_datas += temp[:shot_num]
            self.query_datas += temp[shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]

            self.support_graphs += temp2[:shot_num]
            self.query_graphs += temp2[shot_num:shot_num + query_num]

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels
        self.graphs = self.task.support_graphs if self.split == 'train' else self.task.query_graphs

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")
class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        graph = self.graphs[idx]
        return image, label, graph
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1
def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和query
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)
    print("Draw done")

    return 0


def Draw_Classification_Map(label, name: str, scale: float = 1.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name, format='jpg', transparent=True, dpi=dpi, pad_inches=0)
    pass


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    h, w = gt.shape
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [h, w, class_count])
    return GT_One_Hot


class PatchDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, datalist, graphlist, is_Already_to_gpu=False, device="cpu"):
        # TODO
        self.datalist = datalist
        self.graphlist = graphlist
        self.device = device
        self.on_gpu = is_Already_to_gpu

    def __getitem__(self, index):
        # TODO
        if self.on_gpu == True:
            # do
            sample = self.datalist[index][0]
            label = self.datalist[index][1]
            subgraph = self.graphlist[index]
        else:
            # sample=torch.from_numpy(self.datalist[index][0]).to(self.device)
            sample = self.datalist[index][0].to(self.device)
            label = self.datalist[index][1].to(self.device)
            subgraph = self.graphlist[index].to(self.device)
        # return sample,label,subgraph,subedges
        return sample, label, subgraph
        # 这里需要注意的是，第一步：read one data，是一个data
        pass

    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.datalist)


class GetDataset(object):
    def __init__(self, HSI, extend=2):
        HSI = np.array(HSI, np.float32)
        self.extend = extend
        self.H, self.W, self.B = HSI.shape
        self.HSI = np.pad(HSI, ((extend, extend), (extend, extend), (0, 0)), mode='constant')
        self.a = 1

    def getAllSamples(self, gt: np.array, rotation: int = 0, convert2tensor: bool = True):
        if len(gt.shape) == 3:
            h, w, c = gt.shape
            gtFlag = np.sum(gt, axis=-1, keepdims=False)
        else:
            h, w = gt.shape
            gtFlag = gt
        # 对每一个非0标签样本进行切分
        patch_len = self.extend * 2 + 1
        samples = []
        for i in range(h):
            for j in range(w):
                if gtFlag[i, j] == 0: continue
                # 由于HSI扩充,(i,j)像素在HSI中的实际坐标为(i+self.extend,j+self.extend)
                datacube = self.HSI[i:i + patch_len, j:j + patch_len, :]
                if rotation != 0:
                    M = cv2.getRotationMatrix2D((self.extend, self.extend), rotation, 1)
                    # 第三个参数：变换后的图像大小
                    datacube = cv2.warpAffine(datacube, M, (patch_len, patch_len))

                sample = []
                if convert2tensor:
                    sample.append(torch.from_numpy(datacube))
                    sample.append(gt[i, j])
                else:
                    sample.append(datacube)
                    sample.append(gt[i, j])

                samples.append(sample)

                # plt.show()
        return samples

    def getSamples(self, gt: np.array, rotation: int = 0, convert2tensor: bool = True):
        if len(gt.shape) == 3:
            h, w, c = gt.shape
            gtFlag = np.sum(gt, axis=-1, keepdims=False)
        else:
            h, w = gt.shape
            gtFlag = gt
        # 对每一个非0标签样本进行切分
        patch_len = self.extend * 2 + 1
        samples = []
        for i in range(h):
            for j in range(w):
                if gtFlag[i, j].any() == 0: continue
                # 由于HSI扩充,(i,j)像素在HSI中的实际坐标为(i+self.extend,j+self.extend)
                datacube = self.HSI[i:i + patch_len, j:j + patch_len, :]
                if rotation != 0:
                    M = cv2.getRotationMatrix2D((self.extend, self.extend), rotation, 1)
                    # 第三个参数：变换后的图像大小
                    datacube = cv2.warpAffine(datacube, M, (patch_len, patch_len))

                sample = []
                if convert2tensor:
                    sample.append(torch.from_numpy(datacube))
                    sample.append(torch.from_numpy(gt[i, j]))
                else:
                    sample.append(datacube)
                    sample.append(gt[i, j])

                samples.append(sample)

                # plt.show()
        return samples


def get_Samples_GT(seed: int, gt: np.array, class_count: int, train_ratio, val_ratio, samples_type: str = 'ratio', ):
    # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        train_number_per_class = []
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32') + \
                                     np.ceil(samplesCount * val_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            train_number_per_class.append(np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        val_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = list(train_rand_idx[c])
            train_data_index = train_data_index + a[:train_number_per_class[c]]
            val_data_index = val_data_index + a[train_number_per_class[c]:]

        ##将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        test_data_index = all_data_index - train_data_index - val_data_index

        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    if samples_type == 'same_num':
        if int(train_ratio) == 0 or int(val_ratio) == 0:
            print("ERROR: The number of samples for train. or val. is equal to 0.")
            exit(-1)
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_ratio)  # 每类相同数量样本,则训练比例为每类样本数量
            real_val_samples_per_class = int(val_ratio)  # 每类相同数量样本,则训练比例为每类样本数量

            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class >= samplesCount:
                real_train_samples_per_class = samplesCount - 1
                real_val_samples_per_class = 1
            else:
                real_val_samples_per_class = real_val_samples_per_class if (
                                                                                   real_val_samples_per_class + real_train_samples_per_class) <= samplesCount else samplesCount - real_train_samples_per_class
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class + real_val_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
            if real_val_samples_per_class > 0:
                rand_real_idx_per_class_val = idx[rand_idx[-real_val_samples_per_class:]]
                val_rand_idx.append(rand_real_idx_per_class_val)

        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        val_data_index = []
        for c in range(val_rand_idx.shape[0]):
            a = val_rand_idx[c]
            for j in range(a.shape[0]):
                val_data_index.append(a[j])
        val_data_index = np.array(val_data_index)

        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)

        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        # background_idx = np.where(gt_reshape == 0)[-1]
        # background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - val_data_index

        # # 从测试集中随机选取部分样本作为验证集
        # val_data_count = int(val_samples)  # 验证集数量
        # val_data_index = random.sample(test_data_index, val_data_count)
        # val_data_index = set(val_data_index)

        # test_data_index = 'test_data'_index - val_data_index
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass

    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    return train_samples_gt, test_samples_gt, val_samples_gt


def LabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


def label2edge(label, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = label.size(1)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute edge
    edge = torch.eq(label_i, label_j).float().to(device)
    return edge


def one_hot_encode(num_classes, class_idx, device):
    """
    one-hot encode the ground truth
    :param num_classes: number of total class
    :param class_idx: belonging class's index
    :param device: the gpu device that holds the one-hot encoded ground truth label
    :return: one-hot encoded ground truth label
    """
    return torch.eye(num_classes)[class_idx].to(device)


def preprocess(num_ways, num_shots, num_queries, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots
    num_samples = num_supports + num_queries * num_ways

    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)

    return num_supports, num_samples, query_edge_mask, evaluation_mask


def preprocess_one(num_supports, num_samples, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task

    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)

    return num_supports, query_edge_mask, evaluation_mask


def set_tensors(tensors, batch):
    """
    set data to initialized tensors
    :param tensors: initialized data tensors
    :param batch: current batch of data
    :return: None
    """
    support_data, support_label, query_data, query_label = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    tensors['query_label'].resize_(query_label.size()).copy_(query_label)

def initialize_nodes_edges(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):
    """
    :param batch: data batch
    :param num_supports: number of samples in support set
    :param tensors: initialized tensors for holding data
    :param batch_size: how many tasks per batch
    :param num_queries: number of samples in query set
    :param num_ways: number of classes for each few-shot task
    :param device: the gpu device that holds all data

    :return: data of support set,
             label of support set,
             data of query set,
             label of query set,
             data of support and query set,
             label of support and query set,
             initialized node features of distribution graph (Vd_(0)),
             initialized edge features of point graph (Ep_(0)),
             initialized edge_features_of distribution graph (Ed_(0))
    """
    # allocate data in this batch to specific variables
    set_tensors(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)
    support_label = tensors['support_label'].squeeze(0)
    query_data = tensors['query_data'].squeeze(0)
    query_label = tensors['query_label'].squeeze(0)

    # initialize nodes of distribution graph
    node_gd_init_support = label2edge(support_label, device)
    node_gd_init_query = (torch.ones([batch_size, num_queries, num_supports])
                          * torch.tensor(1. / num_supports)).to(device)
    node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)

    # initialize edges of point graph
    all_data = torch.cat([support_data, query_data], 1)
    all_label = torch.cat([support_label, query_label], 1)
    all_label_in_edge = label2edge(all_label, device)
    edge_feature_gp = all_label_in_edge.clone()

    # uniform initialization for point graph's edges
    edge_feature_gp[:, num_supports:, :num_supports] = 1. / num_supports
    edge_feature_gp[:, :num_supports, num_supports:] = 1. / num_supports
    edge_feature_gp[:, num_supports:, num_supports:] = 0
    for i in range(num_queries):
        edge_feature_gp[:, num_supports + i, num_supports + i] = 1

    # initialize edges of distribution graph (same as point graph)
    edge_feature_gd = edge_feature_gp.clone()

    return support_data, support_label, query_data, query_label, all_data, all_label_in_edge, \
           node_feature_gd, edge_feature_gp, edge_feature_gd


def unlabel2edge(data, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = data.size(1)
    # reshape
    scores = torch.einsum('bhm,bmn->bhn', data, data.transpose(2, 1))
    edge = torch.nn.functional.softmax(scores, dim=-1)
    return edge

def set_tensors_unlabel(tensors, batch):

    support_data, support_label, query_data = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    # tensors['query_label'].resize_(query_label.size()).copy_(query_label)

def initialize_nodes_edges_unlabel(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):
    # allocate data in this batch to specific variables
    set_tensors_unlabel(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)
    support_label = tensors['support_label'].squeeze(0)
    query_data = tensors['query_data'].squeeze(0)
    # query_label = tensors['query_label'].squeeze(0)

    # initialize nodes of distribution graph

    node_gd_init_support = label2edge(support_label, device)
    node_gd_init_query = (torch.ones([batch_size, num_queries, num_supports])
                          * torch.tensor(1. / num_supports)).to(device)
    node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)

    # initialize edges of point graph
    all_data = torch.cat([support_data, query_data], 1)
    all_label_in_edge = unlabel2edge(all_data, device)
    edge_feature_gp = all_label_in_edge.clone()

    # uniform initialization for point graph's edges
    edge_feature_gp[:, num_supports:, :num_supports] = 1. / num_supports
    edge_feature_gp[:, :num_supports, num_supports:] = 1. / num_supports
    edge_feature_gp[:, num_supports:, num_supports:] = 0
    for i in range(num_queries):
        edge_feature_gp[:, num_supports + i, num_supports + i] = 1

    # initialize edges of distribution graph (same as point graph)
    edge_feature_gd = edge_feature_gp.clone()

    return support_data, query_data, all_data, all_label_in_edge, \
           node_feature_gd, edge_feature_gp, edge_feature_gd


def OT(src, tar, ori=False, sub=False, **kwargs):
    wd, gwd = [], []
    for i in range(len(src)):
        source_share, target_share = src[i], tar[i]
        cos_distance = cost_matrix_batch_torch(source_share, target_share)
        cos_distance = cos_distance.transpose(1, 2)
        # TODO: GW as graph matching loss
        beta = 0.1
        if sub:
            cos_distance = kwargs['w_st'] * cos_distance
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)

        wd_val = - IPOT_distance_torch_batch_uniform(cos_dist, source_share.size(0), source_share.size(2),
                                                     target_share.size(2), iteration=30)
        gwd_val = GW_distance_uniform(source_share, target_share, sub, **kwargs)
        wd.append(abs(wd_val))
        gwd.append(abs(gwd_val))

    ot = sum(wd) / len(wd) + sum(gwd) / len(gwd)
    return ot, sum(wd) / len(wd), sum(gwd) / len(gwd)

def allocate_tensors():
    """
    init data tensors
    :return: data tensors
    """
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    tensors['query_label'] = torch.LongTensor()
    return tensors