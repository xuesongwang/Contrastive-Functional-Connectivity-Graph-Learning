# Copyright (c) 2022 Xuesong Wang
# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>


import os
import warnings
import glob
import csv
import re
import numpy as np
import scipy.io as sio
from nilearn import connectome
warnings.filterwarnings("ignore")

root_folder = '/share/scratch/xuesongwang/nilearn_data/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')


def fetch_filenames(subject_IDs, file_type, atlas):
    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types
        filemapping  : resulting file name format
    returns:
        filenames    : list of filetypes (same length as subject_list)
    """

    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_' + atlas: '_rois_' + atlas + '.1D'}
    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)
        try:
            try:
                os.chdir(data_folder)
                filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
            except:
                os.chdir(data_folder + '/' + subject_IDs[i])
                filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            filenames.append('N/A')
    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name, silence=False, data_folder = data_folder):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        if silence != True:
            print("Reading timeseries file %s" % fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


#  compute connectivity matrices
def subject_connectivity(timeseries, subjects, atlas_name, kind, iter_no='', seed=1234,
                         n_subjects='', save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['TPE', 'TE', 'correlation','partial correlation']:
        if kind not in ['TPE', 'TE']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)

    if save:
        if kind not in ['TPE', 'TE']:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(save_path, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity
        else:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(save_path, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
                                                iter_no) + '_' + str(seed) + '_' + validation_ext + str(
                                                n_subjects) + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity_fit

def subject_singleview_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]
        subject_file = os.path.join(save_path, subject,
                                subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity

def subject_multiview_connectivity(timeseries, subject, atlas_name, kind, n_view = 3, save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    # split time series
    ts_split = np.array_split(timeseries, n_view)
    seires_dict = {}
    for view in range(n_view):
        connectivity = []
        if kind in ['tangent', 'partial correlation', 'correlation']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform([ts_split[view]])[0]
            seires_dict['view'+str(view)+'_connectivity'] = connectivity
    subject_file = os.path.join(save_path, subject,
                                subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_multiview.mat')
    sio.savemat(subject_file, seires_dict)
    return connectivity


# Get the list of subject IDs

def get_ids(num_subjects=None):
    """
    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                if score == 'HANDEDNESS_CATEGORY':
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 'R'
                    elif row[score] == 'Mixed':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    elif row[score] == 'L->R':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    else:
                        scores_dict[row['SUB_ID']] = row[score]
                elif (score == 'FIQ' or score == 'PIQ' or score == 'VIQ'):
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 100
                    else:
                        scores_dict[row['SUB_ID']] = float(row[score])

                else:
                    scores_dict[row['SUB_ID']] = row[score]

    return scores_dict

def get_subject_aux(subject_list, aux_list):
    import pandas as pd
    scores_dict = {}
    handedness_dict = {'L': 0, 'R': 1, 'Ambi': 2}
    site_keys = np.array(['CALTECH','CMU','KKI','LEUVEN_1','LEUVEN_2','MAX_MUN','NYU','OHSU','OLIN','PITT',
                 'SBL','SDSU','STANFORD','TRINITY','UCLA_1','UCLA_2','UM_1','UM_2','USM','YALE'])

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = dict()
                for score in aux_list:
                    if score == 'HANDEDNESS_CATEGORY':
                        if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                            temp = 'R'
                        elif row[score] == 'Mixed':
                            temp = 'Ambi'
                        elif row[score] == 'L->R':
                            temp = 'Ambi'
                        else:
                            temp = row[score]
                        scores_dict[row['SUB_ID']][score] = handedness_dict[temp]
                    elif (score == 'FIQ' or score == 'PIQ' or score == 'VIQ'):
                        if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                            scores_dict[row['SUB_ID']][score] = 100
                        else:
                            scores_dict[row['SUB_ID']][score] = float(row[score])
                    elif (score == 'AGE_AT_SCAN' or score == 'SEX'):
                        scores_dict[row['SUB_ID']][score] = float(row[score])
                    elif score == 'SITE_ID':
                        scores_dict[row['SUB_ID']][score] = np.argwhere(site_keys == row[score])[0,0]
                    else:
                        scores_dict[row['SUB_ID']][score] = row[score]
    # temp = pd.DataFrame(scores_dict).transpose()
    # print(np.unique(temp['SITE_ID']))
    return scores_dict


# preprocess phenotypes. Categorical -> ordinal representation
def preprocess_phenotypes(pheno_ft, params):
    if params['model'] == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')

    return (pheno_ft)


# create phenotype feature vector to concatenate with fmri feature vectors
def phenotype_ft_vector(pheno_ft, num_subjects, params):
    gender = pheno_ft[:, 0]
    if params['model'] == 'MIDA':
        eye = pheno_ft[:, 0]
        hand = pheno_ft[:, 2]
        age = pheno_ft[:, 3]
        fiq = pheno_ft[:, 4]
    else:
        eye = pheno_ft[:, 2]
        hand = pheno_ft[:, 3]
        age = pheno_ft[:, 4]
        fiq = pheno_ft[:, 5]

    phenotype_ft = np.zeros((num_subjects, 4))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))

    for i in range(num_subjects):
        phenotype_ft[i, int(gender[i])] = 1
        phenotype_ft[i, -2] = age[i]
        phenotype_ft[i, -1] = fiq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1

    if params['model'] == 'MIDA':
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
    else:
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

    return phenotype_ft


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                 variable='connectivity', data_folder =data_folder):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        fl = os.path.join(data_folder, subject,
                              subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")


        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)

    if kind in ['TE', 'TPE']:
        norm_networks = [mat for mat in all_networks]
    else:
        norm_networks = [np.arctanh(mat) for mat in all_networks]

    networks = np.stack(norm_networks)

    return networks


def get_multiview_networks(subject_list, kind, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                 variable='connectivity', data_folder =data_folder):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        fl = os.path.join(data_folder, subject,
                              subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + "_multiview.mat")


        matrix = sio.loadmat(fl)
        views = []
        for viewname in matrix.keys():
            if 'view' not in viewname:
                continue
            views.append(matrix[viewname])
        views = np.array(views)
        all_networks.append(views)

    if kind in ['TE', 'TPE']:
        norm_networks = [mat for mat in all_networks]
    else:
        norm_networks = [np.arctanh(mat) for mat in all_networks]

    networks = np.stack(norm_networks)

    return networks