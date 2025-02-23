from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
criterion = F.cross_entropy
mse_loss = nn.MSELoss()

from .base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Reba_CLient(Client):
    def __init__(self, options, id, local_dataset, system_attr,  ):

        super(Reba_CLient, self).__init__(options, id, local_dataset, system_attr, )


    def local_train(self, round_i, global_proto):
        begin_time = time.time()
        # 得到原型！！！

        # 得到模型参数
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, round_i, global_proto)
        local_model_protos = self.get_local_proto(self.local_dataset, self.options, )
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats, (self.local_data_class_distribution, local_model_protos)
 
    def local_update(self, local_dataset, options, round_i, global_protos):
        # global_protos = self.global_proto
        if options['batch_size'] == 100:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                feature, pred = self.model(X)
                # compute proximal_term
                loglikelihood = torch.log(self.balanced_softmax1(pred))
                # print(loglikelihood.shape)  # Should be [batch_size, num_classes]
                # print(y.shape)  # Should be [batch_size]
                loss0 = nn.NLLLoss()(loglikelihood, y)
                
                loss1 = 0
                if 0.5 > 0 and global_protos and round_i > 0:
                    protos_new = feature.clone().detach()
                    features = feature.clone().detach()
                    protos_aug = feature.clone().detach()
                    labels_agu = y.clone().detach()
                    label = y.clone().detach()
                    for i in range(len(y)):
                        yi = y[i].item()
                        label = i % options['num_calsses']
                        if (label in global_protos) and (yi in global_protos):
                            protos_aug[i] = 1.0 * (global_protos[label] - global_protos[yi]) + protos_new[i]
                            labels_agu[i] = label
                    g_logits = self.model.feature2logit(protos_aug)
                    # py = self.prior_y_batch(labels_agu)
                    loglikelihood = torch.log(self.balanced_softmax2(g_logits, self.local_data_class_distribution))
                    loss1 = nn.NLLLoss()(loglikelihood, labels_agu)
                loss = loss0 + 1 * loss1

                # loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
            local_model_paras = self.get_flat_model_params()

        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict   

    def get_local_proto(self, local_dataset, options, ):

        localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        local_protos_list = {}
        for X, y in localTrainDataLoader:
            if self.gpu >= 0:
                X, y = X.cuda(), y.cuda()
            feature, pred = self.model(X)
            protos = feature.clone().detach()
            for i in range(len(y)):
                if y[i].item() in local_protos_list.keys():
                    local_protos_list[y[i].item()].append(protos[i,:])
                else:
                    local_protos_list[y[i].item()] = [protos[i,:]]
        # print("local_protos_list", local_protos_list)
        local_protos = get_protos(local_protos_list)
        # print("local_protos", local_protos)
        return local_protos
    
    def prior_y_batch(self, labels):
        py = torch.zeros(self.options['num_calsses'])
        total = len(labels)
        for i in range(self.options['num_calsses']):
            py[i] = py[i] + (i == labels).sum()
        py = py/(total)
        # print(total,py)
        py = py.to(self.device)
        return py
    
    def balanced_softmax1(self, logit):
        py = torch.tensor(self.local_data_class_distribution).cuda()
        # print(py)
        exp = torch.exp(logit)
        eps = 1e-3 # self.eps # 1e-3 if (self.avail_class > 3) else 1e-2
        py1 = (1 - eps) * py + eps / self.options['num_calsses'] # 0.01 for 2 class and 0.001 for 3 class, 0.0001 for more class
        py_smooth = py1 / (py1.sum())
        pc_exp = exp * (py_smooth)
        pc_sftmx = pc_exp / (pc_exp.sum(dim=1).reshape((-1, 1)) + 1e-8)
        return pc_sftmx
    
    def balanced_softmax2(self, logit, py):
        py = torch.tensor(self.local_data_class_distribution).cuda()
        exp = torch.exp(logit)
        eps = 1e-3 #self.args.eps # 1e-3 if (self.avail_class > 3) else 1e-2
        py1 = (1 - eps ) * py + eps / self.options['num_calsses']  # 0.01 for 2 class and 0.001 for 3 class, 0.0001 for more class
        py_smooth = py1 / (py1.sum())
        pc_exp = exp * (py_smooth)
        pc_sftmx = pc_exp / (pc_exp.sum(dim=1).reshape((-1, 1))+1e-8)
        return pc_sftmx  
    
def get_protos(protos):
    protos_mean = {}
    for [label, proto_list] in protos.items():
        proto = 0 * proto_list[0]
        for i in proto_list:
            proto += i
        protos_mean[label] = proto / len(proto_list)

    return protos_mean

