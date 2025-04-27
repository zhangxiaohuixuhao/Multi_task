import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# import torch.functional as F
import torch.nn.functional as F
import utility.utils as utils
from utility.utils import oect_data_proc_std
from utility.dvs_dataset import DvsTFDataset
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import time
from ann import ann_model
'''OPTION'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--device_name', type=str, default='data_cell3')
parser.add_argument('--device_cnt', type=int, default=1)

parser.add_argument('--feat_path', type=str, default='11222014_final_012cls') # 4 class version
parser.add_argument('--log_dir', type=str, default='')
options = parser.parse_args()

num_cls = 4
spike = 4
img_width = 28

'''PATH'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')

DEVICE_DIR = os.path.join(ROOT_DIR, 'data')

SAVE_PATH = os.path.join(ROOT_DIR, 'log/dvs')
time_str = datetime.now().strftime('%m%d%H%M')
savepath = os.path.join(SAVE_PATH, f'{options.log_dir}{time_str}')

for path in [SAVE_PATH, savepath]:
    if not os.path.exists(path):
        os.mkdir(path)

'''load dataset'''

'''load device data'''
device_path = os.path.join(DEVICE_DIR, f'{options.device_name}.xlsx')
device_output = oect_data_proc_std(path=device_path,
                                   device_test_cnt=options.device_cnt)
device_output = device_output.to_numpy().astype(np.float32)

'''define model'''
# model = nn.Sequential(nn.Linear(in_features=img_width ** 2, out_features=num_cls))
model = ann_model.model(img_width ** 2, num_cls,
                            options.batch, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=options.lr,weight_decay=0.001)
cross_entropy = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

def feat_extract(savepath=''):
    '''feature extraction'''
    print('Extracting feature')
    data = np.load('dataset/dvs_data_32frame/train_data_{}.npy'.format(spike))
    tr_label = np.load('dataset/dvs_data_32frame/train_lab_{}.npy'.format(spike)) 
    tr_lab = np.zeros((len(tr_label), (tr_label.max()+1)))
    for i in range(len(tr_label)):
        tr_lab[i, tr_label[i]] = 1
    data = torch.tensor(data)
    num_tr_data = data.shape[0]
    data = data.view(num_tr_data, -1, img_width ** 2)
    tr_feat = utils.batch_rc_feat_extract(data, device_output, options.device_cnt,
                                                spike, num_tr_data)
    tr_lab = torch.tensor(tr_lab)

    data = np.load('dataset/dvs_data_32frame/test_data_{}.npy'.format(spike))
    label = np.load('dataset/dvs_data_32frame/test_lab_{}.npy'.format(spike))
    te_lab = np.zeros((len(label), (label.max()+1)))
    for i in range(len(label)):
        te_lab[i, label[i]] = 1
    data = torch.tensor(data)
    num_te_data = data.shape[0]
    data = data.view(num_te_data, -1, img_width ** 2)
    te_feat = utils.batch_rc_feat_extract(data, device_output, options.device_cnt,
                                                spike, num_te_data)
    te_lab = torch.tensor(te_lab)
    # te_feat = torch.cat(te_feat, dim=0)
    tr_feat = tr_feat.view(num_tr_data, 1, img_width ** 2)
    te_feat = te_feat.view(num_te_data, 1, img_width ** 2)
    feat = torch.cat((tr_feat, te_feat),dim=0)
    feat = (feat - feat.mean()) / feat.std()
    tr_feat = feat[:num_tr_data, ...]
    te_feat = feat[num_tr_data:, ...]

    if savepath:
        torch.save((tr_feat, tr_lab), os.path.join(savepath, 'train_feat_{}.pt'.format(spike)))
        torch.save((te_feat, te_lab), os.path.join(savepath, 'test_feat_{}.pt'.format(spike)))
        
    return num_tr_data, num_te_data, tr_feat, te_feat, tr_label

def test(num_data,
         num_class,
         batchsize,
         test_loader,
         model,
         criterion
         ):
    # test
    te_accs = []
    te_losses = 0
    te_outputs = []
    targets = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):

            this_batch_size = len(data)

            data = data.to(torch.float)

            output = F.softmax(model(data.squeeze()), dim=-1)
            loss = criterion(output, target)
            te_outputs.append(output)
            acc = torch.sum(output.argmax(dim=-1) == target.argmax(dim=-1) ) / this_batch_size
            te_accs.append(acc)
            te_losses += loss.cpu().numpy()
            targets.append(target.argmax(dim=-1))
        te_acc = (sum(te_accs) * batchsize / num_data).numpy()
        te_loss = te_losses / num_data

        # log infos
        log = "test acc: %.6f" % te_acc
        print(log)

        if batchsize == 1:
            te_outputs = torch.stack(te_outputs, dim=0)
        else:
            te_outputs = torch.cat(te_outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        # confusion matrix
        conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=-1))

        conf_mat_dataframe = pd.DataFrame(conf_mat,
                                        index=list(range(num_class)),
                                        columns=list(range(num_class)))

        conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)

        return te_acc, te_loss, conf_mat, conf_mat_normalized


# first try load feature
try:
    tr_feat, te_feat = torch.load(f'log/dvs/{options.feat_path}/feat.pt')
    print('Use extracted feature')
except:
    print('No existing feature, extract feature from dataset')
    num_tr_data, num_te_data, tr_feat, te_feat, tr_lab = feat_extract(savepath=savepath)

tr_dataset = DvsTFDataset(os.path.join(savepath, 'train_feat_{}.pt'.format(spike)))
te_dataset = DvsTFDataset(os.path.join(savepath, 'test_feat_{}.pt'.format(spike)))

tr_loader = DataLoader(tr_dataset, batch_size=options.batch, shuffle=True, num_workers=0)
te_loader = DataLoader(te_dataset, batch_size=options.batch, shuffle=False)
criterion = nn.CrossEntropyLoss()
'''training'''
print('start training')
start_time = time.time()
acc_list = []
loss_list = []
log_list = []
test_acc_list = [] 
test_loss_list = []
conf_mat_list = []
for epoch in range(options.epoch):
    acc = []
    loss = 0
    for i, (data, target) in enumerate(tr_loader):
        optimizer.zero_grad()

        data = data.to(torch.float).squeeze()
        # target = torch.tensor(target, dtype=torch.long)
        # readout layer
        logic = F.softmax(model(data), dim=-1)
        batch_loss = criterion(logic, target)
        loss += batch_loss
        batch_acc = torch.sum(logic.argmax(dim=-1) == target.argmax(dim=-1)) / options.batch
        acc.append(batch_acc)
        batch_loss.backward()
        optimizer.step()
    te_acc, te_loss, conf_mat, conf_mat_normalized = test(num_te_data,
                                                            num_cls,
                                                            options.batch,
                                                            te_loader,
                                                            model,
                                                            criterion)
    tr_acc, tr_loss, tr_conf_mat, tr_conf_mat_normalized = test(num_tr_data,
                                                            num_cls,
                                                            options.batch,
                                                            tr_loader,
                                                            model,
                                                            criterion)
    test_acc_list.append(te_acc)
    test_loss_list.append(te_loss)
    if te_acc == max(test_acc_list):
        np.save(os.path.join(savepath, f'test_results.npy'), conf_mat)
        np.save(os.path.join(savepath, f'train_results.npy'), tr_conf_mat)
        # save readout layer
        torch.save(model, os.path.join(savepath, f'{te_acc*1e5:.0f}.pt'))

    scheduler.step()
    acc_epoch = (sum(acc) * options.batch / num_tr_data).numpy()
    acc_list.append(acc_epoch)
    loss_list.append(loss.detach().numpy())

    epoch_end_time = time.time()
    if epoch == 0:
        epoch_time = epoch_end_time - start_time 
    else:
        epoch_time = epoch_end_time - epoch_start_time
    epoch_start_time = epoch_end_time

    # log info
    log = "epoch: %d, loss: %.4f, acc: %.6f, test acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, te_acc, epoch_time)
    print(log)
    log_list.append(log + '\n')
utils.write_log(savepath, log_list)
# save results
os.path.join(savepath, f'{max(test_acc_list)*1e5:.0f}.pt')
np.savez(os.path.join(savepath, f'train_results.npz'), acc_list=acc_list, te_acc_list=test_acc_list, loss_list=loss_list, te_loss_list=test_loss_list,
             conf_mats=conf_mat_list)


# pca
# color = ['coral', 'dodgerblue', 'tan', 'orange', 'green', 'silver', 'chocolate', 'lightblue', 'violet', 'crimson']
# color_list = [color[i] for i in range(num_cls)]
# tr_feat = tr_feat.squeeze(1)
# pca = PCA(n_components=3)
# outputs_pca = pca.fit_transform(tr_feat, 0)
# np.savetxt('pca_3_tr.csv', outputs_pca)
# np.savetxt('label_tr.csv', tr_lab)
# # plt.figure()
# plt.scatter(outputs_pca[:, 0], outputs_pca[:, 1], c=color_list)
# plt.savefig(f'{savepath}/dvs_2cls_pca.pdf')
# plt.close()


# # conf mat
# # text_label = ['arm roll', 'hand clap', 'arm circle', ]
# text_label = list(range(num_cls))
# conf_mat = confusion_matrix(labels, outputs)
# confusion_matrix_df = pd.DataFrame(conf_mat, index=text_label, columns=text_label)
# plt.figure(figsize=(num_cls, num_cls))
# sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap=plt.cm.Blues)
# plt.savefig(f'{savepath}/conf_mat_dvs.pdf', format='pdf')
# plt.close()
# normed_conf_mat = conf_mat / np.expand_dims(conf_mat.sum(1), -1)
# normed_confusion_matrix_df = pd.DataFrame(normed_conf_mat, index=text_label, columns=text_label)
# plt.figure()
# sns.heatmap(normed_confusion_matrix_df, annot=True, fmt='.2f', cmap=plt.cm.Blues)
# plt.savefig(f'{savepath}/normed_conf_mat_dvs.pdf', format='pdf')
# plt.close()
# print('Confusion matrix saved')


# if __name__ == '__main__':
#     pass
