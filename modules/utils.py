import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import confusion_matrix, roc_auc_score
import networkx as nx
import torch
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import os
import seaborn as sns
import matplotlib.pyplot as plt


def print_report(label, site_data):
    # site collumn -8
    # df = pd.DataFrame(columns=['ADHD', 'HC'])
    sites = {1: "PK" , 3:"KKI", 5: "NYU", 7: "PU", 4: "NI", 6: "OSHU"}
    df = dict()
    MULTICLASS = True if np.unique(label).shape[0] >2 else False
    for site in sites.keys():
        site_index = site_data == site
        if sum(site_index) == 0: # skip sites
            continue
        site_label = label[site_index]
        if MULTICLASS:
            ad0 = sum(site_label == 0)
            hc1 = sum(site_label == 1)
            hc2 = sum(site_label == 2)
            df[sites[site]] = {"Subtype 0": ad0, "Subtype 1": hc1, "Subtype 2": hc2}
        else:
            pos_count = sum(site_label)
            neg_count = sum(site_label == 0)
            df[sites[site]] = {"ADHD": int(pos_count.item()), "HC": int(neg_count.item())}
    df = pd.DataFrame(df)
    print(df)


def print_report_adhd(dataset):
    label = []
    sites = []
    for data_subject in dataset:
        label.append(data_subject.y1.cpu().numpy())
        sites.append(data_subject.aux[0].cpu().numpy())
    label = np.array(label)
    sites = np.array(sites)
    print_report(label, sites)


def print_report_abide(label=None, site_data=None, dataset=None):
    import pandas as pd
    # site collumn -8
    # df = pd.DataFrame(columns=['ADHD', 'HC'])
    sites_dict = ['CALTECH', 'CMU', 'KKI', 'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'NYU', 'OHSU', 'OLIN', 'PITT',
              'SBL', 'SDSU', 'STANFORD', 'TRINITY', 'UCLA_1', 'UCLA_2', 'UM_1', 'UM_2', 'USM', 'YALE']
    if dataset is not None:
        label = []
        site_data = []
        for data_subject in dataset:
            label.append(data_subject.y1[0].cpu().numpy())
            site_data.append(data_subject.aux[0].cpu().numpy())
        label = np.array(label)
        site_data = np.array(site_data)
    df = dict()
    for site in np.unique(site_data):
        site_index = site_data == site
        if sum(site_index) == 0: # skip sites
            continue
        site_label = label[site_index]
        pos_count = sum(site_label)
        neg_count = sum(site_label == 0)
        df[sites_dict[int(site)]] = {"Austism": int(pos_count.item()), "HC": int(neg_count.item())}
    df = pd.DataFrame(df)
    df = df.transpose()
    print(df)


def load_unimportant_FC_ind(path='../network_mapping.csv', nroi=90):
    df = pd.read_csv(path)
    network_list = ['VN', 'AN', 'BLN', 'DMN', 'SMN', 'SN', 'MN', 'CCN']
    fc = np.ones((nroi, nroi))

    for network in network_list:
        roi_idx = df[df['Networks'] == network]['Lables'].values - 1

        xidx, yidx = np.meshgrid(roi_idx, roi_idx)
        xidx = xidx.reshape(-1)
        yidx = yidx.reshape(-1)
        fc[xidx, yidx] = 0


    # get off-diagonal elements
    nondiag = np.ones((nroi, nroi)) - np.eye(nroi)
    nondiag_idx = nondiag.nonzero()
    fc_edge = fc[nondiag_idx[0], nondiag_idx[1]]

    indx_edge = fc_edge.nonzero()
    indx_node = fc.reshape(-1).nonzero()
    return indx_edge, indx_node


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k).cpu().numpy()
    V = dist.reshape(M*k).cpu().numpy()
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W.toarray(), W


def get_topk_edge_attributes(sim, k=15, return_sim=False):
    # select the top-k connections from sim
    sim, idx = sim.sort(descending=True)

    sim = sim[:, 1:k + 1] # excluding first element as in self-connection
    idx = idx[:, 1:k + 1]
    sim, adj = adjacency(sim, idx)
    adj = adj.astype(np.float32).tocoo()
    edge_attr = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_attr[i] = sim[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_attr = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_attr))
    edge_index = edge_index.long()
    edge_index, edge_attr = coalesce(edge_index, edge_attr, sim.shape[0], sim.shape[0])
    if return_sim == False:
        return edge_index, edge_attr
    else:
        return edge_index, edge_attr, sim


def preprocess_batch(batch):
    # get a batch of samples, each sample is consisted of 3 views of the same subject
    # randomly sample two views from the sample and form a new batch
    x1 = batch.x1
    edge_index1 = batch.edge_index1
    x1_batch = batch.x1_batch
    edge_attr1 = batch.edge_attr1
    pos1 = batch.pos1
    batch.y1 = batch.y1.reshape(len(pos1), -1)
    y1 = batch.y1.long()
    aux_withsite = batch.aux.reshape(len(pos1), -1)
    aux = aux_withsite[:, 1:]
    site = aux_withsite[:, 0]

    view1 = (x1, edge_index1, x1_batch, edge_attr1, pos1, y1, aux, site)

    x2 = batch.x2
    edge_index2 = batch.edge_index2
    x2_batch = batch.x2_batch
    edge_attr2 = batch.edge_attr2
    pos2 = batch.pos2
    batch.y2 = batch.y2.reshape(len(pos2), -1)
    y2 = batch.y2.long()
    view2 = (x2, edge_index2, x2_batch, edge_attr2, pos2, y2, aux, site)

    x3 = batch.x3
    edge_index3 = batch.edge_index3
    x3_batch = batch.x3_batch
    edge_attr3 = batch.edge_attr3
    pos3 = batch.pos3
    batch.y3 = batch.y3.reshape(len(pos3), -1)
    y3 = batch.y3[:, 0].long()
    view3 = (x3, edge_index3, x3_batch, edge_attr3, pos3, y3, aux, site)

    combine = [[view1, view2],
               [view1, view3],
               [view2, view3]]
    comb_index = np.random.randint(len(combine))
    views = combine[comb_index]
    return views


def accuracy(y, y_predict):
    return sum(y_predict == y) / y.shape[0]


def sensitivity(y, y_predict):
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
    return tp/(tp + fn)


def specificity(y, y_predict):
    tn, fp, fn, tp = confusion_matrix(y, y_predict).ravel()
    return tn/(tn + fp)


def evaluate_knn(clf, data, path ='./results/ContraKNN/clf.csv'):
    mask = data.test_mask
    sites = {1: "PK", 3: "KKI", 4: "NI", 5: "NYU", 6: "OSHU", 7: "PU"}
    df = dict()
    x = data.x[mask].detach().cpu().numpy()
    test_site = data.site[mask].detach().cpu().numpy()
    label = data.y[mask].detach().cpu().numpy()
    for site in sites.keys():
        site_index = test_site == site
        if sum(site_index) == 0:  # no data for the site
            continue

        site_label = label[site_index]
        site_data = x[site_index]

        site_y_pred = clf.predict(site_data)
        acc = accuracy(site_label, site_y_pred)
        sns = sensitivity(site_label, site_y_pred)
        spc = specificity(site_label, site_y_pred)
        auc = roc_auc_score(site_label, clf.predict_proba(site_data)[:, 1])
        df[sites[site]] = {"accuracy": acc,
                           "sensitivity": sns,
                           "specificity": spc,
                           "auc": auc}

    all_y_pred = clf.predict(x)
    acc = accuracy(label, all_y_pred)
    sns = sensitivity(label, all_y_pred)
    spc = specificity(label, all_y_pred)
    auc = roc_auc_score(label, clf.predict_proba(x)[:, 1])
    df['Overall'] = {"accuracy": acc,
                     "sensitivity": sns,
                     "specificity": spc,
                     "auc": auc}

    df = pd.DataFrame(df)
    if path is not None:
        df.to_csv(path)
    print(df)
    return df


def visualize_graph(G, y, seed=1, method_name = 'Raw FC', save_fig = 'CGL', label_list=['HC', 'ADHD']):
    fig, ax = plt.subplots(figsize=(13,13))
    plt.xticks([])
    plt.yticks([])
    # cmaps = ['#00b8ff', '#fb9800']
    # cmaps = ['#bc59cf', '#389565']
    # cmaps = ['#00cb00', '#ff88ff']
    cmaps = ['#ff48c4','#2bd1fc','#f3ea5f','#c04df9', '#ff3f3f',
             '#ff6666', '#c0d6e4', '#468499', '#ccff00', '#3399ff',
             '#0027b2', '#8b4bce', '#eea83d', '#aacbd2', '#f64a4a',
             '#37c5ee', '#f47f20', '#ee392c', '#70be44', '#c3a3cd']
    colors = np.array([cmaps[0]] * y.shape[0])
    for color_i, c in enumerate(np.unique(y)):
        colors[y == c] = cmaps[color_i]
    alpha = 0.7
    postions = nx.spring_layout(G, seed=seed)
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=12345), with_labels=False,
                     node_color=colors, node_size=160, width=1.5, edge_color='gray', alpha=alpha)

    for idx, c in enumerate(np.unique(y)):
        temp_idx = np.argwhere(y == c)[0, 0]
        plt.scatter(postions[temp_idx][0], postions[temp_idx][1], s=200, label=label_list[idx], c=cmaps[idx], alpha=alpha)

    font_size= 42
    plt.legend(loc='upper left', frameon=True, fontsize=font_size)

    # plt.xlabel('CGL(self-supervised)', fontsize=font_size)
    plt.xlabel(method_name, fontsize=font_size)
    # plt.xlabel('CGL+DGC(Ours)', fontsize=font_size)
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(0, 0.01)

    plt.setp(ax.spines.values(), linewidth=3)  # framewidth of the axises
    plt.tight_layout()
    # plt.savefig('CGL.png')
    # plt.savefig('./saved_imgs/%s_seed%s.png'%(save_fig, seed))
    plt.savefig('%s_seed%s.png'%(save_fig, seed))
    plt.show()


def print_subtypes(cluster_labels, subject_ids, clf_labels, root_dir = '/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90/processed'):
    subject_ids = subject_ids.cpu().numpy()
    clf_labels = clf_labels.cpu().numpy()
    cluster_names = ['Subtype1', 'Subtype2', 'Subtype3']
    subject_name_id_map = pd.read_csv(os.path.join(root_dir, "name_ID_mapping.csv"))

    subject_name_list = [subject_name_id_map[subject_name_id_map['ID'] == idx]['name'].values[0] for idx in subject_ids]
    cluster_name_list = [cluster_names[idx] for idx in cluster_labels]
    df = pd.DataFrame({'subtype_index': cluster_labels,
                       'subtype_name': cluster_name_list,
                       'subject_index': subject_ids,
                       'subject_name': subject_name_list,
                       'clf_labels': clf_labels})
    df.to_csv("saved_csv/overall_subtype_subjects.csv", index=False)
    for groupname, df_sub in df.groupby('subtype_name'):
        ad_ratio = 100*df_sub[df_sub['clf_labels']==1].shape[0]/len(df_sub)
        print(groupname, "total number of subjects:%d"%len(df_sub), "AD ratio within the subtype: %.2f%%"%(ad_ratio))


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    df = np.concatenate([h, np.expand_dims(color, axis=-1)], axis=-1)
    df = pd.DataFrame(df, columns=['x1', 'x2', 'aux'])
    sns.scatterplot(data=df, x='x1', y='x2', hue='aux')
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


def draw_view_difference(x1, x2):
    def f(x, temperature = 0.001):
        return x/temperature

    # def normalize_df_by

    # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / 0.1)
    homo_diff = np.mean(x1 * x2, axis=1)
    x = np.concatenate([x1, x2], axis=0)  # 2 x n_subject
    temp = np.matmul(x, x.T)/x1.shape[1]
    mask = 1 - np.tile(np.eye(x1.shape[0]), (2, 2))  # remove diagnal and same view distences
    heter_sim = temp * mask
    heter_sim_idx = heter_sim.nonzero()
    heter_diff = heter_sim[heter_sim_idx]

    # homogeneous difference
    # homo_diff = f(np.mean((x1 - x2)**2, axis=-1)) # n_subject
    # # heterogeneous diff
    # x = np.concatenate([x1, x2], axis=0) # 2 x n_subject
    # x1_heter = np.expand_dims(x, 0)
    # x2_heter = np.expand_dims(x, 1)
    # heter_diff = f(np.mean((x1_heter - x2_heter)**2, axis=-1)) # 2n_sub x 2n_sub
    # print('finished')
    # mask = 1 - np.tile(np.eye(x1.shape[0]), (2 , 2)) # remove diagnal and same view distences
    # heter_diff = heter_diff * mask
    # heter_diff_idx = heter_diff.nonzero()
    # heter_diff = heter_diff[heter_diff_idx]

    name = ['homo'] * homo_diff.shape[0] + ['heter'] * heter_diff.shape[0]
    data = np.concatenate([homo_diff, heter_diff])
    df = pd.DataFrame(data, columns=['attraction'])
    df['pair'] = name

    # Plot formatting
    fig, ax = plt.subplots(figsize=(20, 13))
    sns.set(font_scale=3, style="whitegrid")  # CRITICAL: you can only set legend fontsize here
    g = sns.histplot(
        df, x="attraction", hue="pair",
        stat="probability", common_norm=False,
        kde=True, fill=True, ax=ax,
        log_scale=(False, True),
        line_kws=dict(linewidth=6)
    )
    font_size = 42
    # plt.xlabel('Raw FC feature attraction', fontsize=font_size)
    plt.xlabel('Contrastive feature attraction', fontsize=font_size)
    plt.ylabel('Probability density function', fontsize=font_size)

    # df['xlabel'] = 'Contrastive Features'
    # ax = sns.violinplot(
    #     x = 'xlabel', y= "attraction",  hue='feature',
    #     data=df, scale_hue=True,
    #     inner='quartile',
    #     scale="area", split=True, palette={'homo': '#00b8ff', 'heter': '#fb9800'}
    # )

    # sns.displot(df, x="difference", hue="name", stat="probability")

    # sns.kdeplot(df, x="difference", hue="feature", common_norm=False,)

    # g.set_xlim(-100, 100)
    # g.set_ylim(0, 0.01)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    g.tick_params(axis='x', rotation=9) # rotate x axis
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    # plt.savefig('raw_feature.png')
    plt.savefig('contra_attraction.png')
    plt.show()

def draw_distr_difference(x1, x2):
    def f(x, temperature = 0.001):
        return x/temperature

    # def normalize_df_by

    # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / 0.1)
    homo_diff = np.mean(x1 * x2, axis=1)
    x = np.concatenate([x1, x2], axis=0)  # 2 x n_subject
    temp = np.matmul(x,  x.T)/x1.shape[1]
    mask = 1 - np.tile(np.eye(x1.shape[0]), (2, 2))  # remove diagnal and same view distences
    heter_sim = temp * mask
    heter_sim_idx = heter_sim.nonzero()
    heter_diff = heter_sim[heter_sim_idx]

    # homogeneous difference
    # homo_diff = f(np.mean((x1 - x2)**2, axis=-1)) # n_subject
    # # heterogeneous diff
    # x = np.concatenate([x1, x2], axis=0) # 2 x n_subject
    # x1_heter = np.expand_dims(x, 0)
    # x2_heter = np.expand_dims(x, 1)
    # heter_diff = f(np.mean((x1_heter - x2_heter)**2, axis=-1)) # 2n_sub x 2n_sub
    # print('finished')
    # mask = 1 - np.tile(np.eye(x1.shape[0]), (2 , 2)) # remove diagnal and same view distences
    # heter_diff = heter_diff * mask
    # heter_diff_idx = heter_diff.nonzero()
    # heter_diff = heter_diff[heter_diff_idx]

    name = ['homo'] * homo_diff.shape[0] + ['heter'] * heter_diff.shape[0]
    data = np.concatenate([homo_diff, heter_diff])
    df = pd.DataFrame(data, columns=['attraction'])
    df['pair'] = name

    # Plot formatting
    fig, ax = plt.subplots(figsize=(20, 13))
    sns.set(font_scale=3, style="whitegrid")  # CRITICAL: you can only set legend fontsize here
    g = sns.boxplot(x="pair", y="attraction",
                     data=df, linewidth=3, showfliers = False)

    # g = sns.histplot(
    #     df, x="attraction", hue="pair",
    #     stat="probability", common_norm=False,
    #     kde=True, fill=True, ax=ax,
    #     log_scale=(False, True),
    #     line_kws=dict(linewidth=6)
    # )
    font_size = 42
    # plt.xlabel('Raw FC feature', fontsize=font_size)
    plt.xlabel('Contrastive feature', fontsize=font_size)
    plt.ylabel('Feature attraction', fontsize=font_size)

    # df['xlabel'] = 'Contrastive Features'
    # ax = sns.violinplot(
    #     x = 'xlabel', y= "attraction",  hue='feature',
    #     data=df, scale_hue=True,
    #     inner='quartile',
    #     scale="area", split=True, palette={'homo': '#00b8ff', 'heter': '#fb9800'}
    # )

    # sns.displot(df, x="difference", hue="name", stat="probability")

    # sns.kdeplot(df, x="difference", hue="feature", common_norm=False,)

    # g.set_xlim(-100, 100)
    # g.set_ylim(0, 0.01)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # g.tick_params(axis='x', rotation=9) # rotate x axis
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    # plt.savefig('raw_feature.png')
    plt.savefig('contra_attraction.png')
    plt.show()



if __name__ == '__main__':
    # to test a certain function
    load_unimportant_FC_ind()