"""
07-2022
Author: XueSong Wang <xuesong.wang1@unsw.edu.au>
University of New South Wales
"""
import time
import copy
import os
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TopKPooling, ChebConv, DynamicEdgeConv, GlobalAttention
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops, to_networkx
from torch_sparse import spspmm
from modules.utils import *
from modules.losses import FocalLoss, compute_contrastive_loss
from sklearn.cluster import spectral_clustering
from sklearn.manifold import TSNE
import networkx as nx
import numpy as np


class ContraGraphLearning(torch.nn.Module):
    def __init__(self, indim, ratio, R=200, auxdim=0, device=torch.device("cpu")):
        '''
        :param indim: (int) node feature dimension,
        :param ratio: (float) pooling ratio in (0,1)
        :param R: (int) number of ROIs
        :param auxdim: (int) number of PCD features
        '''
        super(ContraGraphLearning, self).__init__()
        self.device = device
        self.indim = indim
        self.latent_dim = 64
        self.auxdim = auxdim
        self.R = R
        self.conv1 = ChebConv(indim, self.latent_dim, K=3)
        self.pool1 = TopKPooling(self.latent_dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.conv2 = ChebConv(self.latent_dim, self.latent_dim, K=3)
        self.pool2 = TopKPooling(self.latent_dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.out_dim = 512
        self.feature_dim = 256
        self.fc_encoder = torch.nn.Linear((self.latent_dim + self.latent_dim) * 2, self.feature_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.feature_dim)
        self.contra_encoder = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim//4, bias=False),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_dim//4, self.out_dim))

    def graphconv(self, x, edge_index, batch, edge_attr, x_aux=None):
        if x_aux is not None:
            x_aux = x_aux.unsqueeze(1).repeat(1, self.R, 1)
            x_aux = x_aux.reshape(x.shape[0], -1)
            x = torch.cat([x, x_aux], dim=1)

        x = self.conv1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = torch.cat([x1,x2], dim=1)

        feature = self.bn1(F.relu(self.fc_encoder(x)))
        # feature = F.dropout(feature, p=0.5, training=self.training) # not sure if it's useful
        out = self.contra_encoder(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def forward(self, batch, return_option='out', explain=False, explain_input_mask=None):
        """
        batch: Multiview data
        return_option: 'x', 'edge', 'out'
        explain_input_mask: mask the input features, could be a node feature mask or
                                   edge feature mask for feature importance, not graph prediction
        """
        data_view1, data_view2 = preprocess_batch(batch)
        x1, edge_index1, x1_batch, edge_attr1, pos1, y1, aux1, site1 = data_view1
        x2, edge_index2, x2_batch, edge_attr2, pos2, y2, aux2, site2 = data_view2
        if return_option == 'x':
            if explain:
                x1 = x1*explain_input_mask
                x2 = x2*explain_input_mask
            x1.requires_grad_(True)
            x2.requires_grad_(True)
        elif return_option == 'edge':
            if explain:
                n_sample = edge_attr1.shape[0] // explain_input_mask.shape[0]
                edge_attr1 = explain_input_mask.unsqueeze(0).repeat(n_sample, 1).reshape(-1).contiguous()
                edge_attr2 = explain_input_mask.unsqueeze(0).repeat(n_sample, 1).reshape(-1).contiguous()
            edge_attr1.requires_grad_(True)
            edge_attr2.requires_grad_(True)
        feature1, out1 = self.graphconv(x1, edge_index1, x1_batch, edge_attr1, x_aux=aux1)
        feature2, out2 = self.graphconv(x2, edge_index2, x2_batch, edge_attr2, x_aux=aux2)
        if return_option == 'x':
            return x1, x2, feature1, feature2, y1, site1, x1_batch
        elif return_option == 'edge':
            return edge_attr1, edge_attr2, feature1, feature2, y1, site1, x1_batch
        else:
            return out1, out2, feature1, feature2, y1, site1


class DynamicGraphClassification(torch.nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(DynamicGraphClassification, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edgeconv_dim = 64
        self.nn1 = nn.Sequential(nn.Linear(in_dim*2, self.edgeconv_dim, bias=False), # x2 cuz edgeconv needs self.nn([x_i, x_i - x_j])
                                nn.BatchNorm1d(self.edgeconv_dim),
                                nn.LeakyReLU(negative_slope=0.2))
        self.edge_conv1 = DynamicEdgeConv(self.nn1, k=20, aggr='mean', num_workers=4)
        self.nn2 = nn.Sequential(nn.Linear(self.edgeconv_dim*2, self.edgeconv_dim, bias=False),
                                 nn.BatchNorm1d(self.edgeconv_dim),
                                 nn.LeakyReLU(negative_slope=0.2))
        self.edge_conv2 = DynamicEdgeConv(self.nn2, k=20, aggr='mean', num_workers=4)
        self.nn3 = nn.Sequential(nn.Linear(self.edgeconv_dim*2, self.edgeconv_dim, bias=False),
                                 nn.BatchNorm1d(self.edgeconv_dim),
                                 nn.LeakyReLU(negative_slope=0.2))
        self.edge_conv3 = DynamicEdgeConv(self.nn3, k=20, aggr='mean', num_workers=4)
        self.concat_dim = 512
        self.dim1 = 256
        self.dim2 = 128
        self.concat_fc = nn.Linear(self.edgeconv_dim*3, self.concat_dim, bias=False)
        self.mlp1 = nn.Sequential(nn.Linear(self.concat_dim, self.dim1),
                                 nn.BatchNorm1d(self.dim1),
                                 nn.LeakyReLU(negative_slope=0.2))
        self.mlp2 = nn.Sequential(nn.Linear(self.dim1, self.dim2),
                                  nn.BatchNorm1d(self.dim2),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.clf = nn.Linear(self.dim2, self.out_dim)

    def forward(self, x):
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x = torch.cat([x1, x2, x3], dim=-1)
        # x = torch.cat([x1, x2], dim=-1)
        x = self.concat_fc(x)
        x = self.mlp1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        feature = self.mlp2(x)
        x = F.dropout(feature, p=0.5, training=self.training)
        x = self.clf(x)
        return x, feature


def knn_baseline(data, k = 20):
    from sklearn.neighbors import KNeighborsClassifier
    clf_knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4)
    mask = data.train_mask
    clf_knn.fit(data.x[mask].detach().cpu().numpy(), data.y[mask].detach().cpu().numpy())
    return clf_knn


def mask_onehot2number(data):
    val_mask = data.val_mask.int()
    test_mask = data.test_mask.int()
    val_mask[val_mask == 1] = 2
    test_mask[test_mask == 1] = 3
    mask = data.train_mask.int() + val_mask + test_mask
    return mask.cpu().numpy()-1


class Model:
    def __init__(self, input_dim, nclass, nroi, auxdim, topk_ratio=0.5,
                 optimizer='Adam', lr_encoder=0.001, lr_clf=0.001, weightdecay=5e-3, stepsize=20, gamma=0.5,
                 device=torch.device("cuda:0"), save_path = './save_path', seed=123):
        self.device = device
        self.cgl_encoder = ContraGraphLearning(input_dim, topk_ratio, R=nroi, auxdim=auxdim, device=device).double().to(device)
        self.dgc_clf = DynamicGraphClassification(self.cgl_encoder.feature_dim * 2, nclass,).double().to(device)
        self.init_optimizer(optimizer, lr_encoder, lr_clf, weightdecay, stepsize, gamma)
        self.seed = seed
        self.init_save_path(save_path)
        self.clf_loss = FocalLoss(alpha=5, gamma=2, reduction='mean')

    def print_model(self):
        print(self.cgl_encoder)
        print(self.dgc_clf)

    def init_save_path(self, save_path, encoder_save_path=None, clf_save_path=None):
        self.save_path = save_path
        if encoder_save_path is None:
            self.encoder_save_path = os.path.join(self.save_path, 'CGL_seed%s.pth'%self.seed)
        else:
            self.encoder_save_path = os.path.join(self.save_path, encoder_save_path)
        if clf_save_path is None:
            self.clf_save_path = os.path.join(self.save_path, 'DGC_seed%s.pth'%self.seed)
        else:
            self.clf_save_path = os.path.join(self.save_path, clf_save_path)

    def init_optimizer(self, opt_name, lr_encoder, lr_clf, weightdecay, stepsize, gamma):
        if opt_name == 'Adam':
            self.opt_encoder = torch.optim.Adam(self.cgl_encoder.parameters(), lr=lr_encoder, weight_decay=weightdecay)
            self.opt_clf = torch.optim.Adam(self.dgc_clf.parameters(), lr=lr_clf, weight_decay=weightdecay)
        elif opt_name == 'SGD':
            self.opt_encoder = torch.optim.SGD(self.cgl_encoder.parameters(), lr=lr_encoder, momentum=0.9,
                                                weight_decay=weightdecay, nesterov=True)
            self.opt_clf = torch.optim.SGD(self.dgc_clf.parameters(), lr=lr_clf, momentum=0.9,
                                                 weight_decay=weightdecay, nesterov=True)
        self.scheduler_encoder = lr_scheduler.StepLR(self.opt_encoder, step_size=stepsize, gamma=gamma)
        self.scheduler_clf = lr_scheduler.StepLR(self.opt_clf, step_size=stepsize, gamma=gamma)

    def load_cgl_encoder(self):
        self.cgl_encoder.load_state_dict(
            torch.load(self.encoder_save_path, map_location=self.device))

    def load_dgc_clf(self):
        self.dgc_clf.load_state_dict(
            torch.load(self.clf_save_path, map_location=self.device))

    def train_encoder_one_epoch(self, train_loader):
        self.cgl_encoder.train()
        loss_all = 0
        step = 0
        for batch in train_loader:
            self.opt_encoder.zero_grad()
            batch = batch.to(self.device)
            out1, out2, feature1, feature2, y, site = self.cgl_encoder(batch)
            # loss = supercontra(torch.cat([out1.unsqueeze(dim=1), out2.unsqueeze(dim=1)],dim=1), labels=y)
            loss = compute_contrastive_loss(out1, out2)
            step = step + 1
            loss.backward()
            loss_all += loss.item() * y.shape[0]
            self.opt_encoder.step()

        print('train...........')
        for param_group in self.opt_encoder.param_groups:
            print("LR encoder", param_group['lr'])
        self.scheduler_encoder.step()
        return loss_all / len(train_loader.dataset)

    def train_cgl_encoder(self, train_loader, val_loader, max_epoch=150, save_model=False, load_model=False):
        if load_model:
            self.load_cgl_encoder()
            print("Succesfully load CGL encoder!")
            return
        best_loss = 1e10
        best_epoch = 0
        for epoch in range(0, max_epoch):
            since = time.time()
            tr_loss = self.train_encoder_one_epoch(train_loader)
            val_loss = self.test_encoder(val_loader)
            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s , best epoch {:d}'.format(time_elapsed // 60, time_elapsed % 60, best_epoch))
            print('Epoch: {:03d}, Train Loss: {:.7f}, Test Loss: {:.7f}'.format(epoch, tr_loss, val_loss))

            if val_loss < best_loss and epoch > 5:
                print("saving best encoder")
                best_epoch = epoch
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.cgl_encoder.state_dict())
                if save_model:
                    # torch.save(best_model_wts, os.path.join(opt.save_path, 'CGL_encoder_seed' + str(seed) + '.pth'))
                    torch.save(best_model_wts, self.encoder_save_path)

    def test_encoder(self, loader):
        self.cgl_encoder.eval()
        loss_all = 0
        for batch in loader:
            batch = batch.to(self.device)
            out1, out2, feature1, feature2, y, site = self.cgl_encoder(batch)
            # loss = supercontra(torch.cat([out1.unsqueeze(dim=1), out2.unsqueeze(dim=1)], dim=1), labels=y)
            loss = compute_contrastive_loss(out1, out2)
            loss_all += loss.item() * y.shape[0]
        print('testing........... ')
        return loss_all / len(loader.dataset)

    def clf_preprocessing(self, dataloader_list = [], use_knn=False):
        x1_list = [] # use view1 as node features
        x2_list = [] # use view2 as edge features to compute similarity
        y_list = []
        site_list = []
        subject_id_list = []
        self.cgl_encoder.eval()
        for dataloader in dataloader_list:
            for batch in dataloader:
                batch = batch.to(self.device)
                _, _, feature1, feature2, y, site = self.cgl_encoder(batch)
                x1_list.extend(feature1.detach())
                x2_list.extend(feature2.detach())
                y_list.extend(y)
                subject_id_list.extend(batch.subject_name)
                site_list.extend(site)
        x1 = torch.stack(x1_list)
        y = torch.stack(y_list).long().reshape(-1)
        site = torch.stack(site_list)
        subject_id = torch.stack(subject_id_list).long().reshape(-1)

        x2 = torch.stack(x2_list)
        x = torch.cat([x1, x2], dim=-1)

        if len(dataloader_list)==1: # only a few loader given
            train_count, val_count, test_count = len(dataloader_list[0].dataset), 0, 0
        else:
            train_count, val_count, test_count = len(dataloader_list[0].dataset),  len(dataloader_list[1].dataset), len(dataloader_list[2].dataset)
        train_mask, val_mask, test_mask = torch.zeros(y.shape[0]), torch.zeros(y.shape[0]), torch.zeros(y.shape[0])
        train_mask[:train_count] = 1
        val_mask[train_count: (train_count+val_count)] = 1
        test_mask[(train_count + val_count):] = 1

        if not use_knn:
            # build graph for Dynamic Graph Classification, no edge attr required
            data = Data(x=x, y=y, site=site, subject_id=subject_id,
                        train_mask=train_mask.bool(), val_mask=val_mask.bool(), test_mask=test_mask.bool()).to(self.device)
        else:
            sim_matrix = torch.exp(torch.mm(x2, x2.t().contiguous()) / 0.1)
            edge_index, edge_attr = get_topk_edge_attributes(sim_matrix, k=use_knn)
            data = Data(x=x1, edge_index=edge_index, edge_attr=edge_attr, y=y, site=site, subject_id=subject_id,
                        train_mask=train_mask.bool(), val_mask=val_mask.bool(), test_mask=test_mask.bool())
        # Gather some statistics about the graph.
        # print(f'Finished classifier preprocessing!')
        # print(f'Number of nodes: {data.num_nodes}')
        # print(f'Number of training nodes: {data.train_mask.sum()}')
        # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        return data

    def train_clf_one_epoch(self, data):
        self.dgc_clf.train()

        self.opt_clf.zero_grad()

        y_pred, _ = self.dgc_clf(data.x)
        loss = self.clf_loss(y_pred[data.train_mask], data.y[data.train_mask])

        loss.backward()
        self.opt_clf.step()

        print('train...........')
        self.scheduler_clf.step()

        for param_group in self.opt_clf.param_groups:
            print("DGC classifier", param_group['lr'])
        return loss.item()

    def train_dgc_clf(self, train_loader, val_loader, test_loader, max_epoch=100, save_model=False, load_model=False):
        if val_loader is None:
            datalist = [train_loader]
        else:
            datalist = [train_loader, val_loader, test_loader]
        # build the population graph
        data = self.clf_preprocessing(datalist)
        if load_model:
            self.load_dgc_clf()
            print("Succesfully load DGC classifier!")
            return data

        best_loss = 1e10
        best_epoch = 0
        for epoch in range(0, max_epoch):
            since = time.time()
            tr_loss = self.train_clf_one_epoch(data)
            tr_acc, _ = self.test_clf(data, mode='train')
            val_acc, val_loss = self.test_clf(data, mode='val')
            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s best epoch:{:d}'.format(time_elapsed // 60, time_elapsed % 60, best_epoch))
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                  'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}, '.format(epoch, tr_loss, tr_acc,
                                                                                    val_loss, val_acc))
            if val_loss < best_loss and epoch > 5:
                best_epoch = epoch
                print("saving best classifier")
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.dgc_clf.state_dict())
                if save_model:
                    torch.save(best_model_wts,  self.clf_save_path)
        return data

    def test_clf(self, data, mode='train'):
        self.dgc_clf.eval()
        y_pred, _ = self.dgc_clf(data.x)
        pred = y_pred.argmax(dim=1)  # Use the class with highest probability.
        if mode == 'test':
            test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
            acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
            loss = self.clf_loss(y_pred[data.test_mask], data.y[data.test_mask])
        elif mode == 'val':
            test_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
            acc = int(test_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
            loss = self.clf_loss(y_pred[data.val_mask], data.y[data.val_mask])
        elif mode == 'train':
            test_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
            acc = int(test_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
            loss = self.clf_loss(y_pred[data.train_mask], data.y[data.train_mask])
        return acc, loss.item()

    def train_test_cgl_knn(self, datalist, topk=5):
        data = self.clf_preprocessing(datalist, topk)
        clf_knn = knn_baseline(data)
        print("evaluating CGL + KNN:")
        evaluate_knn(clf_knn, data, path=None)

    def visualize_heter_homo_difference(self, datalist, use_raw_feature=False):
        x1, x2, y, site = self.get_encoder_embedding(datalist, use_raw_feature=use_raw_feature)
        x = TSNE(n_components=2).fit_transform(x1)
        visualize_embedding(x, y)
        visualize_embedding(x, site)
        draw_distr_difference(x1, x2)
        draw_view_difference(x1, x2)

    def evaluation_report(self, data, mode='test', path ='./results/DGC/clf.csv', merge=False,
                          data_option='ADHD'):
        self.dgc_clf.eval()
        y_pred, _ = self.dgc_clf(data.x)
        total_pred = y_pred.argmax(dim=1)  # Use the class with highest probability.
        total_y_prob = F.softmax(y_pred, dim=1)
        if merge == True:
            total_pred[total_pred == 2] = 1  # 2 -> 1
            total_pred = 1 - total_pred  # 0 <->1
            total_y_prob = total_y_prob[:, 0]  # probability of subtype 0 is the prob of real AD
        else:
            total_y_prob = total_y_prob[:, 1]  # probability of class 1
        total_y = data.y
        total_site = data.site
        MULTICLASS = True if len(torch.unique(data.y)) > 2 else False
        if mode == 'test':
            mask = data.test_mask
        elif mode == 'val':
            mask = data.val_mask
        elif mode == 'train':
            mask = data.train_mask.cpu().numpy()
        total_pred = total_pred[mask].detach().cpu().numpy()
        total_y_prob = total_y_prob[mask].detach().cpu().numpy()
        total_y = total_y[mask].cpu().numpy()
        total_site = total_site[mask].cpu().numpy()

        if data_option == 'ADHD':
            sites = {1: "PK", 3: "KKI", 4: "NI", 5: "NYU", 6: "OSHU", 7: "PU"}
        else:
            sites = {0:'CALTECH', 1:'CMU',2: 'KKI', 3:'LEUVEN_1', 4:'LEUVEN_2',
                          5:'MAX_MUN', 6:'NYU', 7:'OHSU', 8:'OLIN', 9:'PITT',
                          10:'SBL', 11:'SDSU', 12:'STANFORD', 13: 'TRINITY', 14:'UCLA_1',
                          15:'UCLA_2', 16:'UM_1', 17:'UM_2', 18:'USM', 19:'YALE'}
        df = dict()
        for site in sites.keys():
            site_index = total_site == site
            if sum(site_index) == 0:  # no data
                continue

            site_label = total_y[site_index]
            site_y_pred = total_pred[site_index]
            site_y_prob = total_y_prob[site_index]
            acc = accuracy(site_label, site_y_pred)

            if MULTICLASS:
                df[sites[site]] = {"accuracy": acc}
                cm = confusion_matrix(site_label, site_y_pred)
                print(cm)
            else:
                sns = sensitivity(site_label, site_y_pred)
                spc = specificity(site_label, site_y_pred)
                auc = roc_auc_score(site_label, site_y_prob)
                df[sites[site]] = {"accuracy": acc,
                                   "sensitivity": sns,
                                   "specificity": spc,
                                   "auc": auc}

        acc = accuracy(total_y, total_pred)
        if MULTICLASS:
            df['Overall'] = {"accuracy": acc}
            cm = confusion_matrix(total_y, total_pred)
            print(cm)
            # save_confusion_matrix(cm, acc)
        else:
            sns = sensitivity(total_y, total_pred)
            spc = specificity(total_y, total_pred)
            auc = roc_auc_score(total_y, total_y_prob)
            df['Overall'] = {"accuracy": acc,
                             "sensitivity": sns,
                             "specificity": spc,
                             "auc": auc}
        df = pd.DataFrame(df)
        if path is not None:
            df.to_csv(path)
        print(df.transpose())
        return df


    def build_population_edgegraph(self, data, k=5):
        _, features = self.dgc_clf(data.x)
        features = features.detach()
        sim_matrix = torch.exp(torch.mm(features, features.t().contiguous()) / 10) # over 10 to avoid inf values and non-symmetrical sim_matrix
        edge_index, edge_attr, sim= get_topk_edge_attributes(sim_matrix, k=k, return_sim=True)

        # build graph
        data = Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, site=data.site, subject_id = data.subject_id,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask )
        # Gather some statistics about the graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        return data, sim

    def visualize_population_graph(self, olddata, seed = 1, choice='label', site=None, data_option='ADHD'):
        data, sim = self.build_population_edgegraph(olddata, k=2)
        G = to_networkx(data, to_undirected=True)
        method_name = 'CGL+DGC'
        site = '' if site is None else site
        if choice == 'label':
            if data_option == 'ADHD':
                label_list = ['HC', 'ADHD']
            else:
                label_list = ['HC', 'Autism']
            visualize_graph(G, y=data.y.cpu().numpy(), method_name=method_name, seed=seed,
                        save_fig='Population Graph s%s_%s'%(site, choice),  label_list=label_list)
        elif choice == 'tr_val_status':
            train_val_tesk_mask = mask_onehot2number(data)
            label_list = ['train', 'val', 'test']
            visualize_graph(G, y=train_val_tesk_mask, method_name=method_name,seed=seed,
                            save_fig='Population Graph s%s_%s'%(site, choice),  label_list=label_list)
        elif choice == 'site':
            if data_option == 'ADHD':
                label_list = ['PKU', 'KKI', 'NYU']
            else:
                label_list = ['CALTECH', 'CMU', 'KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'NYU', 'OHSU', 'OLIN', 'PITT',
                              'SBL', 'SDSU', 'STANFORD', 'TRINITY', 'UCLA_1', 'UCLA_2', 'UM_1', 'UM_2', 'USM', 'YALE']
            visualize_graph(G, y=data.site.cpu().numpy(), label_list=label_list, seed=seed,
                            method_name=method_name, save_fig='Population Graph s%s_%s'%(site, choice))
        elif choice == 'subtype':
            sim = np.asarray(nx.to_numpy_matrix(G))
            cluster_labels = spectral_clustering(sim, n_clusters=3,)
            label_list = ['Type1', 'Type2', 'Type3']
            visualize_graph(G, y=cluster_labels, label_list=label_list, seed=seed,
                            method_name=method_name, save_fig='Population Graph s%s_%s'%(site, choice))
            return cluster_labels

    def get_encoder_embedding(self, loader_list, use_raw_feature=False):
        print('extracting...........')
        self.cgl_encoder.eval()
        x1_list, x2_list, y_list, site_list = [],[],[],[]
        subject_id_list = []
        self.cgl_encoder.eval()
        for dataloader in loader_list:
            for batch in dataloader:
                batch = batch.to(self.device)
                x1, x2, feature1, feature2, y, site, x_batch = self.cgl_encoder(batch, return_option='x')
                if not use_raw_feature:
                    x1_list.extend(feature1.detach())
                    x2_list.extend(feature2.detach())
                else:
                    x1_list.extend(x1.detach())
                    x2_list.extend(x2.detach())
                y_list.extend(y)
                subject_id_list.extend(batch.subject_name)
                site_list.extend(site)
        x1 = torch.stack(x1_list).cpu().numpy()
        x2 = torch.stack(x2_list).cpu().numpy()
        y = torch.stack(y_list).long().reshape(-1).cpu().numpy()
        site = torch.stack(site_list).cpu().numpy()
        return x1, x2, y, site


