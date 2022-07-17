# Copyright (c) 2019 Mwiza Kunda
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
if you do not want to denerate multi-view dataset, you can run main(),
main_multiview() will split the time series into n_views to generate FC matrices
'''

import sys
import os.path as osp
import torch
from os import listdir
import argparse
import pandas as pd
import numpy as np
import deepdish as dd
import warnings
import os
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
from preprocessing import Reader
from modules.loader import Data, MultiviewData

warnings.filterwarnings("ignore")
root_folder = '/share/scratch/xuesongwang/nilearn_data/'
data_folder = os.path.join(root_folder, 'ADHD200/AAL90/')

# Process boolean command line arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                 'MIDA is used to minimize the distribution mismatch between ADHD200 sites')
    parser.add_argument('--atlas', default='aal',
                        help='Atlas for network construction (node definition) options: aal.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')


    args = parser.parse_args()
    print('Arguments: \n', args)


    params = dict()

    params['seed'] = args.seed  # seed for random initialisation

    # Algorithm choice
    params['atlas'] = args.atlas  # Atlas for network construction
    atlas = args.atlas  # Atlas for network construction (node definition)

    # get dataset

    atlas = 'aal'
    # Get subject IDs and class labels
    # get training/testing subject_IDs
    label_train = pd.read_csv(os.path.join(data_folder, 'train_label.csv'))
    label_test = pd.read_csv(os.path.join(data_folder, 'test_label.csv'))
    label = pd.concat([label_train, label_test], axis=0)

    subject_IDs = label['subjectID'].to_list()

    # drop the subjects from unwanted sites
    aux_list = ['Site', 'Age', 'Gender', 'Handedness', 'IQ_Measure', 'Verbal_IQ', 'Performance_IQ', 'Full4_IQ']
    aux = label[aux_list].values
    aux = np.nan_to_num(aux, -999)
    y = label['DX'].values

    # Number of subjects and classes for binary classification
    num_classes = args.nclass
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    # 1 is autism, 0 is control
    y_data = np.zeros([num_subjects, num_classes]) # n x 2

    # Get onehot encoding
    y_data[y == 1, 1] = 1
    y_data[y == 0, 0] = 1

    # Compute feature vectors (vectorised connectivity networks)
    fea_corr = Reader.get_networks(subject_IDs, iter_no='', kind='correlation', atlas_name=atlas, data_folder=data_folder) #(1035, 200, 200)
    fea_pcorr = Reader.get_networks(subject_IDs, iter_no='', kind='partial correlation', atlas_name=atlas, data_folder=data_folder) #(1035, 200, 200)

    if not os.path.exists(os.path.join(data_folder,'raw')):
        os.makedirs(os.path.join(data_folder,'raw'))
    for i, subject in enumerate(subject_IDs):
        dd.io.save(os.path.join(data_folder,'raw',subject+'.h5'),{'corr':fea_corr[i],'pcorr':fea_pcorr[i],'label':y[i]%2,
                                                                  'aux': aux[i]})


def main_multiview():
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                 'MIDA is used to minimize the distribution mismatch between ADHD200 sites')
    parser.add_argument('--atlas', default='aal',
                        help='Atlas for network construction (node definition) options: aal.')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')


    args = parser.parse_args()
    print('Arguments: \n', args)


    params = dict()

    params['seed'] = args.seed  # seed for random initialisation

    # Algorithm choice
    params['atlas'] = args.atlas  # Atlas for network construction
    atlas = args.atlas  # Atlas for network construction (node definition)

    # get dataset

    atlas = 'aal'
    # Get subject IDs and class labels
    # get training/testing subject_IDs
    label_train = pd.read_csv(os.path.join(data_folder, 'train_label.csv'))
    label_test = pd.read_csv(os.path.join(data_folder, 'test_label.csv'))
    label = pd.concat([label_train, label_test], axis=0)

    subject_IDs = label['subjectID'].to_list()

    # drop the subjects from unwanted sites
    aux_list = ['Site', 'Age', 'Gender', 'Handedness', 'IQ_Measure', 'Verbal_IQ', 'Performance_IQ', 'Full4_IQ']
    aux = label[aux_list].values
    aux = np.nan_to_num(aux, -999)
    y = label['DX'].values

    # Number of subjects and classes for binary classification
    num_classes = args.nclass
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    # 1 is autism, 0 is control
    y_data = np.zeros([num_subjects, num_classes]) # n x 2

    # Get onehot encoding
    y_data[y == 1, 1] = 1
    y_data[y == 0, 0] = 1

    # Compute feature vectors (vectorised connectivity networks)
    fea_corr = Reader.get_multiview_networks(subject_IDs, iter_no='', kind='correlation', atlas_name=atlas, data_folder=data_folder) #(1035, 200, 200)
    fea_pcorr = Reader.get_multiview_networks(subject_IDs, iter_no='', kind='partial correlation', atlas_name=atlas, data_folder=data_folder) #(1035, 200, 200)

    if not os.path.exists(os.path.join(data_folder,'raw_multiview')):
        os.makedirs(os.path.join(data_folder,'raw_multiview'))
    for i, subject in enumerate(subject_IDs):
        dd.io.save(os.path.join(data_folder,'raw_multiview',subject+'.h5'),{'corr':fea_corr[i],'pcorr':fea_pcorr[i],'label':y[i]%2,
                                                                  'aux': aux[i]})


def read_single_multiview_data(data_dir,filename, subject_name):
    """ create multiview object from a single subject"""
    temp = dd.io.load(os.path.join(data_dir, filename))

    n_view = temp['pcorr'].shape[0]

    datanivew = []

    # read edge and edge attribute
    pcorr_nview = np.abs(temp['pcorr'][()])
    att_nview = temp['corr'][()]
    pcorr_nview[pcorr_nview == float('inf')] = 0
    att_nview[att_nview == float('inf')] = 0
    label = temp['label'][()]
    y_torch = torch.from_numpy(np.array(label))  # classification

    if 'aux' in temp.keys():
        aux = torch.from_numpy(np.nan_to_num(temp['aux'], -999))

    for view in range(n_view):
        pcorr = pcorr_nview[view]
        num_nodes = pcorr.shape[0]
        G = from_numpy_matrix(pcorr)
        A = nx.to_scipy_sparse_matrix(G)
        adj = A.tocoo()
        edge_att = np.zeros(len(adj.row))
        for i in range(len(adj.row)):
            edge_att[i] = pcorr[adj.row[i], adj.col[i]]

        edge_index = np.stack([adj.row, adj.col])
        edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
        edge_index = edge_index.long()
        edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                        num_nodes)
        att = att_nview[view]
        att_torch = torch.from_numpy(att).float()
        pos = np.diag(np.ones(num_nodes))
        data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att, pos=pos)
        datanivew.append(data)
    # edge_attr: (8010,) = 90x90-90(self loop) so kind of fully connection
    data1 = datanivew[0]
    data2 = datanivew[1]
    data3 = datanivew[2]

    data = MultiviewData(x1=data1.x, edge_index1=data1.edge_index.long(), y1=data1.y, edge_attr1=data1.edge_attr, pos1=data1.pos,
                         x2=data2.x, edge_index2=data2.edge_index.long(), y2=data2.y, edge_attr2=data2.edge_attr, pos2=data2.pos,
                         x3=data3.x, edge_index3=data3.edge_index.long(), y3=data3.y, edge_attr3=data3.edge_attr, pos3=data3.pos,
                         aux=aux, subject_name=subject_name)
    return data


def read_multiview_data(data_dir='/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90/raw_multiview',
                        root_dir='/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90/processed'):
    '''
    Preprocessing the subject_ID.h5 files into dataset.pt
        :param data_dir: (str) directory where subject_ID.h5 files are stored
        :param root_dir: (str) directory where the result dataset.pt is stored
    '''
    from tqdm import tqdm
    import timeit

    onlyfiles = [f for f in listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    onlyfiles.sort()
    start = timeit.default_timer()

    # Subject_NameID = pd.DataFrame({'ID': np.arange(len(onlyfiles)), 'name': [f.split('.')[0] for f in onlyfiles]})
    # Subject_NameID.to_csv(os.path.join(root_dir, "name_ID_mapping.csv"), index=False)

    data = []
    for subject_id, f in enumerate(tqdm((onlyfiles))):
        data.append(read_single_multiview_data(data_dir, f, subject_id))
    torch.save(data, os.path.join(root_dir, 'multiviewdata.pt'))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return data


if __name__ == '__main__':
    # main()
    # main_multiview()
    name = 'ADHD'
    path = '/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90'
    raw_dir = os.path.join(path, 'raw_multiview')
    # load_multiview_site_data(site='P+K')
    read_multiview_data(raw_dir)