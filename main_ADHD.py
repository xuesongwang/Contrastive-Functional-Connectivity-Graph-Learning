"""
07-2022
Author: XueSong Wang <xuesong.wang1@unsw.edu.au>
University of New South Wales
"""

import argparse
import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from modules.loader import train_val_test_split_adhd, MultiviewData
from modules.utils import print_report_adhd
from modules.losses import FocalLoss
from modules.model import Model


seed = 12345
torch.manual_seed(seed)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of CGL encoder training')
parser.add_argument('--n_epochs_clf', type=int, default=100, help='number of epochs of DGC classifier training')
parser.add_argument('--batchSize', type=int, default=100, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/share/scratch/xuesongwang/nilearn_data/ADHD200/AAL90/processed', help='root directory of the dataset')
parser.add_argument('--lr', type = float, default=0.001, help='learning rate for CGL encoder')
parser.add_argument('--lr_clf', type = float, default=0.005, help='learning rate for DGC classifier')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--indim', type=int, default=97, help='feature dim, #ROIs + #clinical features')
parser.add_argument('--nroi', type=int, default=90, help='num of ROIs')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_encoder', type=bool, default=True)
parser.add_argument('--load_clf', type=bool, default=True)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./saved_model/', help='path to save model')
opt = parser.parse_args()



#################### Parameter Initialization #######################
dataroot = opt.dataroot
name = 'ADHD'
opt.save_path = os.path.join(opt.save_path, name)
save_model = opt.save_model
opt_method = opt.optim
num_epoch = opt.n_epochs
num_epoch_clf = opt.n_epochs_clf
if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

################## Define Dataloader ##################################
# to generate 'multiview_datasubtype.pt', go to preprocessing/adhd_02-process_data.py and run read_multiview_data()
dataset = torch.load(os.path.join(dataroot, 'multiviewdata.pt'))
site_choice = None
# site_list = ['P', 'K', 'N', 'P+K', 'P+N', 'K+N']
# site_choice = site_list[5]
# dataset = load_multiview_site_data(site=site_choice)

# train/test/val all using the same sites, could be the whole dataset, or just one site for site-transferability verification
train_id, val_id, test_id = train_val_test_split_adhd(n_sub=len(dataset),dataset=dataset, seed=seed)
train_dataset = [dataset[i] for i in train_id]
val_dataset = [dataset[i] for i in val_id]
test_dataset = [dataset[i] for i in test_id]


print("training set")
print_report_adhd(train_dataset)
print("validating set")
print_report_adhd(val_dataset)
print("testing set")
print_report_adhd(test_dataset)

all_loader = DataLoader(dataset, batch_size=opt.batchSize, follow_batch=['x1', 'x2', 'x3'],shuffle= True)
train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, follow_batch=['x1', 'x2', 'x3'],shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, follow_batch=['x1', 'x2', 'x3'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, follow_batch=['x1', 'x2', 'x3'], shuffle=False)


###############  define loss function ###############
cross_entropy = FocalLoss(alpha=5, gamma=2, reduction='mean')

############### Define Graph Deep Learning Network ##########################
model = Model(opt.indim, opt.nclass, opt.nroi, auxdim=7, topk_ratio=opt.ratio,
              optimizer=opt_method, lr_encoder=opt.lr, lr_clf=opt.lr_clf,
              weightdecay=opt.weightdecay, stepsize=opt.stepsize, gamma=opt.gamma,
              device=device, save_path=opt.save_path,
              seed=seed)
model.print_model()
# You can try with our best model
model.init_save_path(save_path='./saved_model/'+name, encoder_save_path='CGL_best_encoder.pth',
                    clf_save_path='DGC_best_classifier.pth')


if __name__ == '__main__':
    model.train_cgl_encoder(train_loader=train_loader, val_loader=val_loader, max_epoch=opt.n_epochs,
                            save_model=opt.save_model, load_model=opt.load_encoder)

    # VAR-1: completely self-supervised: CGL + KNN
    # model.train_test_cgl_knn([train_loader, val_loader, test_loader], topk=2)
    # model.visualize_heter_homo_difference([train_loader, val_loader, test_loader], use_raw_feature=False)

    # our model: CGL + DGC
    data = model.train_dgc_clf(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, max_epoch=opt.n_epochs_clf,
                        save_model=opt.save_model, load_model=opt.load_clf)
    # set val/test loader as None when you only want to test and visualize train_loader
    # data = model.train_dgc_clg(train_loader=test_loader, val_loader=None, test_loader=None, max_epoch=100,
    #                            save_model=True, load_model=opt.load_clf)

    # TASK1: print classification report
    model.evaluation_report(data, mode='test', path=None)

    # TASK2: Visualization on the population graph with four different metrics: label, subtype, site, tr_val_status
    # model.visualize_population_graph(data, choice='label', seed=seed, site=site_choice)
    # subject_id_cluster = model.visualize_population_graph(data, choice='subtype', seed=seed, site=site_choice)
    # model.visualize_population_graph(data, choice='site', seed=seed)
    # model.visualize_population_graph(data, choice='tr_val_status', seed=seed)