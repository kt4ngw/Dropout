from torch.utils.data import DataLoader, RandomSampler
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

import queue
from src.optimizers.gd import GD
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
from src.cost import ClientAttr, Cost
import math
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
from src.models.model import choose_model

class Async_CLient(Client):
    def __init__(self, options, id, local_dataset, system_attr, FedAsync):
        
        super(Async_CLient, self).__init__(options, id, local_dataset, system_attr)
        self.server = FedAsync
        self.response_queue = queue.Queue()  # 接收服务器信息的队列
        self.client_epoch = None # 全局模型时间戳


    def local_train(self, ):
        while True:
            while self.response_queue.empty():
                time.sleep(1)
            # 从服务器接收权重
            response = self.response_queue.get()
            if response == "STOP":
                break
            else:
                self.local_model_para = response[0]
                self.client_epoch = response[1]
            self.set_flat_model_params(self.local_model_para)
            local_model_paras, return_dict = self.local_update(self.local_dataset, self.options)
            self.server.data_queue.put((self, local_model_paras, self.client_epoch))


    def local_update(self, local_dataset, options, ):
        gobal_model = copy.deepcopy(self.server.get_flat_model_params())
        self.optimizer = GD(self.model.parameters(), lr=options['lr'])
        # batch_size=options['batch_size']
        if options['batch_size'] == -1:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
        else:
            if len(local_dataset) < options['batch_size']:
                localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
            else:
                sampler = RandomSampler(local_dataset, replacement=False, num_samples=1 * options['batch_size'])
                localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)
                # localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                self.optimizer.zero_grad()
                _, pred = self.model(X)
                a = time.time()
                reg_loss = torch.norm(self.get_flat_model_params() - gobal_model, p=2) ** 2
                b = time.time()
                reg_loss = (0.0001 / 2) * reg_loss
                loss = criterion(pred, y) + reg_loss
                loss.backward()
                self.optimizer.step()

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
    
