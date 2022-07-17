'''
Define the MultiviewData object, dataset splitting and preprocessing of the subject.h5 files
'''

import numpy as np
import pandas as pd
import torch
import os
from torch_geometric.data import Data
from torch.utils.data import Dataset


def train_val_test_split_adhd(n_sub=None, dataset=None, seed = 12345):
    """split the datset into training/validation/testing sets, used for main_*.py"""
    np.random.seed(seed)
    if n_sub is None:
        path = os.path.join('/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90', 'processed', 'mlp_data.pt')
        dataset = torch.load(path)
        n_sub = len(dataset[0])
        label = dataset[2].cpu().numpy()
        sites = dataset[3][:, 0].cpu().numpy()
    else:
        label = []
        sites = []
        for data_subject in dataset:
            label.append(data_subject.y1.cpu().numpy())
            sites.append(data_subject.aux[0].cpu().numpy())
        label = np.array(label)
    id = np.arange(n_sub)

    train_id = []
    val_id = []
    test_id = []
    # making sure each site has some data of both classes
    for site in np.unique(sites):
        site_index = sites == site
        label_site = label[site_index]
        id_site = id[site_index]
        if site == 7 or site ==3 or site==6:  # too few ADHD patients in sites 3 and 7 (only 4), manually split
            pos_index = np.argwhere(label_site == 1)[:, 0]
            neg_index = np.argwhere(label_site == 0)[:, 0]
            np.random.shuffle(pos_index)
            np.random.shuffle(neg_index)
            pos_index = id_site[pos_index]
            neg_index = id_site[neg_index]
            ten_percent = int(0.1 * neg_index.shape[0])
            if site == 7:
                train_index = np.concatenate([pos_index[:-2], neg_index[:7*ten_percent]])
                val_index = np.concatenate([pos_index[[-2]], neg_index[7*ten_percent: 8*ten_percent]])
                test_index = np.concatenate([pos_index[[-1]], neg_index[8*ten_percent:]])
            else:
                ten_percent_pos = int(0.1*pos_index.shape[0])
                train_index = np.concatenate([pos_index[:7*ten_percent_pos], neg_index[:7*ten_percent]])
                val_index = np.concatenate([pos_index[7*ten_percent_pos : 8*ten_percent_pos], neg_index[7 * ten_percent: 8 * ten_percent]])
                test_index = np.concatenate([pos_index[8*ten_percent_pos:], neg_index[8 * ten_percent:]])
        else:
            np.random.shuffle(id_site)
            ten_percent = int(0.1 * id_site.shape[0])
            train_index = id_site[:7 * ten_percent]
            val_index = id_site[7 * ten_percent: 8 * ten_percent]
            test_index = id_site[8 * ten_percent:]

        train_id.extend(train_index)
        val_id.extend(val_index)
        test_id.extend(test_index)

    return train_id, val_id, test_id


def train_val_split_multiview(n_sub=None, dataset=None, seed = 12345):
    """split the datset into train and validation sets, used for main_*.py"""
    np.random.seed(seed)
    if n_sub is None:
        path = os.path.join('/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90', 'processed', 'mlp_data.pt')
        dataset = torch.load(path)
        n_sub = len(dataset[0])
        label = dataset[2].cpu().numpy()
        sites = dataset[3][:, 0].cpu().numpy()
    else:
        label = []
        sites = []
        for data_subject in dataset:
            label.append(data_subject.y1.cpu().numpy())
            sites.append(data_subject.aux[0].cpu().numpy())
        label = np.array(label)
    id = np.arange(n_sub)

    train_id = []
    val_id = []
    # making sure each site has some data of both classes
    for site in np.unique(sites):
        site_index = sites == site
        label_site = label[site_index]
        id_site = id[site_index]
        if site == 7 or site ==3 or site==6:  # too few ADHD patients in sites 3 and 7 (only 4), manually split
            pos_index = np.argwhere(label_site == 1)[:, 0]
            neg_index = np.argwhere(label_site == 0)[:, 0]
            np.random.shuffle(pos_index)
            np.random.shuffle(neg_index)
            pos_index = id_site[pos_index]
            neg_index = id_site[neg_index]
            ten_percent = int(0.1 * neg_index.shape[0])
            if site == 7:
                train_index = np.concatenate([pos_index[:-2], neg_index[:8*ten_percent]])
                val_index = np.concatenate([pos_index[[-2]], neg_index[8*ten_percent: ]])
            else:
                ten_percent_pos = int(0.1*pos_index.shape[0])
                train_index = np.concatenate([pos_index[:8*ten_percent_pos], neg_index[:8*ten_percent]])
                val_index = np.concatenate([pos_index[8*ten_percent_pos:], neg_index[8 * ten_percent:]])
        else:
            np.random.shuffle(id_site)
            ten_percent = int(0.1 * id_site.shape[0])
            train_index = id_site[:8 * ten_percent]
            val_index = id_site[8 * ten_percent: ]

        train_id.extend(train_index)
        val_id.extend(val_index)

    return train_id, val_id


def train_val_test_split_abide(n_sub=None, dataset=None, seed=12345, drop_sites=[]):
    np.random.seed(seed)
    label = []
    sites = []
    for data_subject in dataset:
        label.append(data_subject.y1.cpu().numpy())
        sites.append(data_subject.aux[0].cpu().numpy())
    label = np.array(label)
    sites = np.array(sites)

    id = np.arange(n_sub)

    train_id = []
    val_id = []
    test_id = []
    site_dict = {0: 'CALTECH', 1: 'CMU', 2: 'KKI', 3: 'LEUVEN_1', 4: 'LEUVEN_2',
             5: 'MAX_MUN', 6: 'NYU', 7: 'OHSU', 8: 'OLIN', 9: 'PITT',
             10: 'SBL', 11: 'SDSU', 12: 'STANFORD', 13: 'TRINITY', 14: 'UCLA_1',
             15: 'UCLA_2', 16: 'UM_1', 17: 'UM_2', 18: 'USM', 19: 'YALE'}
    # making sure each site has some data of both classes
    for site in np.unique(sites):
        if site_dict[site] in drop_sites:
            print("drop site:", site_dict[site])
            continue
        site_index = sites == site
        label_site = label[site_index]
        id_site = id[site_index]

        pos_index = np.argwhere(label_site == 1)[:, 0]
        neg_index = np.argwhere(label_site == 0)[:, 0]
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)
        pos_index = id_site[pos_index]
        neg_index = id_site[neg_index]
        ten_percent = 0.1 * neg_index.shape[0]
        ten_percent_pos = 0.1 * pos_index.shape[0]
        train_index = np.concatenate([pos_index[: int(7 * ten_percent_pos)], neg_index[: int(7 * ten_percent)]])
        val_index = np.concatenate([pos_index[int(7 * ten_percent_pos): max(int(8 * ten_percent_pos), int(7 * ten_percent_pos)+1)],
                                    neg_index[int(7 * ten_percent): max(int(8 * ten_percent), int(7 * ten_percent)+1)]])
        test_index = np.concatenate([pos_index[max(int(8 * ten_percent_pos), int(7 * ten_percent_pos)+1):],
                                     neg_index[max(int(8 * ten_percent), int(7 * ten_percent)+1):]])

        # np.random.shuffle(id_site)
        # ten_percent = int(0.1 * id_site.shape[0])
        # train_index = id_site[:int(np.ceil(7 * ten_percent))]
        # val_index = id_site[int(np.ceil(7 * ten_percent)): int(np.ceil(8 * ten_percent))]
        # test_index = id_site[int(np.ceil(8 * ten_percent)):]

        train_id.extend(train_index)
        val_id.extend(val_index)
        test_id.extend(test_index)
    return train_id, val_id, test_id



class MultiviewData(Data):
    def __init__(self, x1=None, edge_index1=None, y1=None, edge_attr1=None, pos1=None,
                 x2=None, edge_index2=None, y2=None, edge_attr2=None, pos2 = None,
                 x3=None, edge_index3=None, y3=None, edge_attr3=None, pos3 = None, aux=None, subject_name=None):

        super(MultiviewData, self).__init__()
        self.x1 = x1
        self.edge_index1 = edge_index1
        self.y1 = y1
        self.edge_attr1 = edge_attr1
        self.pos1 = pos1
        self.x2 = x2
        self.edge_index2 = edge_index2
        self.y2 = y2
        self.pos2 = pos2
        self.edge_attr2 = edge_attr2
        self.x3 = x3
        self.edge_index3 = edge_index3
        self.y3 = y3
        self.edge_attr3 = edge_attr3
        self.pos3 = pos3
        self.aux = aux
        self.subject_name = subject_name

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        if key == 'edge_index3':
            return self.x3.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class nonbatchDataset(Dataset):
    def __init__(self, dataset):
        super(nonbatchDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        def preprocess_batch(batch):
            # get a batch of samples, each sample is consisted of 3 views of the same subject
            # randomly sample two views from the sample and form a new batch
            x1 = batch.x1
            edge_index1 = batch.edge_index1
            edge_attr1 = batch.edge_attr1
            y1 = batch.y1.long()
            aux_withsite = batch.aux
            aux = aux_withsite[1:]
            site = aux_withsite[0]

            view1 = (x1, edge_index1, edge_attr1, y1, aux, site)

            x2 = batch.x2
            edge_index2 = batch.edge_index2
            edge_attr2 = batch.edge_attr2
            y2 = batch.y2.long()
            view2 = (x2, edge_index2, edge_attr2, y2, aux, site)

            x3 = batch.x3
            edge_index3 = batch.edge_index3
            edge_attr3 = batch.edge_attr3
            y3 = batch.y3.long()
            view3 = (x3, edge_index3, edge_attr3, y3, aux, site)

            combine = [[view1, view2],
                       [view1, view3],
                       [view2, view3]]
            comb_index = np.random.randint(len(combine))
            views = combine[comb_index]
            return views
        return preprocess_batch(self.dataset[item])
