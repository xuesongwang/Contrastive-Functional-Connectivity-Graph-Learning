# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
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
from nilearn import datasets
import argparse
from preprocessing import Reader
import os
import sys
import pandas as pd
import numpy as np

# Input data variables
code_folder = os.getcwd()

# Input data variables
root_folder = '/share/scratch/xuesongwang/nilearn_data/'
data_folder = os.path.join(root_folder, 'ADHD200/AAL90/')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
# shutil.copyfile(os.path.join(root_folder,'subject_ID.txt'), os.path.join(data_folder, 'subject_IDs.txt'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_adhd():
    data = pd.DataFrame()
    ADHD_path = '/home/xuesongwang/scratch/fMRI_data/ADHD200'
    data_path = 'ADHD200_training_'
    for i in range(8):
        data_chunk = pd.read_csv(os.path.join(ADHD_path, data_path + str(i)+'.csv'))
        data = pd.concat([data, data_chunk], axis =0, ignore_index = True)
    print("load train data success!")
    test_data = pd.read_csv(os.path.join(ADHD_path, 'ADHD200_testing.csv'))
    label = pd.read_csv(os.path.join(ADHD_path, 'phenotypic_train.csv'))
    label['subject_ID'] = np.arange(label.shape[0])
    test_label = pd.read_csv(os.path.join(ADHD_path, 'phenotypic_test.csv'))
    return {"train_data": data, "train_label": label, "test_data": test_data, "test_label": test_label}


def subject_connectivity(data, label, atlas, corr='correlation'):
    feature_list = ['ROI_'+str(i+1) for i in range(90)]
    # new_label = pd.DataFrame() # there are some subjects in label whose fMRI are not provided
    for i, data_subject in data.groupby('subject_ID'):
        subject_id = data_subject['subject_ID'].values[0]

        # get subject_name
        subject_name = label[label['subject_ID'] == subject_id]['subjectID'].values[0]
        subject_folder = os.path.join(data_folder, subject_name)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)
        # new_label = pd.concat([new_label, label[label['subject_ID']==subject_id]])
        # generate correlation matrix create folder and save the matrix
        # Reader.subject_connectivity(data_subject[feature_list].values, subject_name, 'aal', 'correlation', save_path=data_folder)
        Reader.subject_singleview_connectivity(data_subject[feature_list].values, subject_name, atlas, corr, save_path=data_folder)
    return label


def subject_connectivity_multiview(data, label, atlas, corr='correlation'):
    feature_list = ['ROI_'+str(i+1) for i in range(90)]
    # new_label = pd.DataFrame() # there are some subjects in label whose fMRI are not provided
    for i, data_subject in data.groupby('subject_ID'):
        subject_id = data_subject['subject_ID'].values[0]

        # get subject_name
        subject_name = label[label['subject_ID'] == subject_id]['subjectID'].values[0]
        subject_folder = os.path.join(data_folder, subject_name)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)
        # new_label = pd.concat([new_label, label[label['subject_ID']==subject_id]])
        # generate correlation matrix create folder and save the matrix
        # Reader.subject_connectivity(data_subject[feature_list].values, subject_name, 'aal', 'correlation', save_path=data_folder)
        Reader.subject_multiview_connectivity(data_subject[feature_list].values, subject_name, atlas, corr, n_view=3, save_path=data_folder)
    return label


def main():
    data_dict = get_adhd()
    atlas = 'aal'

    # get training/testing subject_IDs
    subject_IDs_train = data_dict['train_label']['subjectID'].to_list()
    subject_IDs_test = data_dict['test_label']['subjectID'].to_list()

    # drop the subjects from unwanted sites

    # training set
    # Compute and save connectivity matrices

    data_dict['train_label'] = subject_connectivity(data_dict['train_data'], data_dict['train_label'], atlas, 'correlation')
    data_dict['train_label'] = subject_connectivity(data_dict['train_data'], data_dict['train_label'], atlas,
                                                    'partial correlation')

    # testing set
    data_dict['test_label'] = subject_connectivity(data_dict['test_data'], data_dict['test_label'], atlas, 'correlation')
    data_dict['test_label'] = subject_connectivity(data_dict['test_data'], data_dict['test_label'], atlas,
                                                   'partial correlation')

def main_multiview():
    data_dict = get_adhd()
    atlas = 'aal'

    # get training/testing subject_IDs
    subject_IDs_train = data_dict['train_label']['subjectID'].to_list()
    subject_IDs_test = data_dict['test_label']['subjectID'].to_list()

    # drop the subjects from unwanted sites

    # training set
    # Compute and save connectivity matrices

    data_dict['train_label'] = subject_connectivity_multiview(data_dict['train_data'], data_dict['train_label'], atlas,
                                                    'correlation')
    data_dict['train_label'] = subject_connectivity_multiview(data_dict['train_data'], data_dict['train_label'], atlas,
                                                    'partial correlation')

    # testing set
    data_dict['test_label'] = subject_connectivity_multiview(data_dict['test_data'], data_dict['test_label'], atlas,
                                                   'correlation')
    data_dict['test_label'] = subject_connectivity_multiview(data_dict['test_data'], data_dict['test_label'], atlas,
                                                   'partial correlation')


if __name__ == '__main__':
    # main()
    main_multiview()