from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import copy
from src.fed_client.async_client import Async_CLient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import threading
import queue
import time
lock = threading.Lock()


class FedAsync(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        # self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(FedAsync, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model)
        # 初始化数据队列，用于服务器与客户端之间传递数据
        self.data_queue = queue.Queue()
        
        # 创建客户端线程列表
        self.client_threads = []
        for client in self.clients:
            client_thread = threading.Thread(target=client.local_train)
            self.client_threads.append(client_thread)

        # 创建服务器线程
        self.server_thread = threading.Thread(target=self.server_run)
        self.epoch = 0 # 全局迭代次数
        self.client_epoch = 0 # 客户端的模型时间戳

    def train(self):
        # print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))

        # 启动服务器线程
        self.server_thread.start()

        # 启动客户端线程
        for thread in self.client_threads:
            thread.start()

        # 等待服务器线程完成
        self.server_thread.join()

        # 等待所有客户端线程完成
        for thread in self.client_threads:
            thread.join()

    def server_run(self):
        print('=== Select {} clients per round ===\n'.format(int(self.clients_num)))
        seq = []
        # 给参数给本地   
        for client in self.clients:
            client.response_queue.put((copy.deepcopy(self.latest_global_model), self.epoch)) 
        # self.test_latest_model_on_testdata(0)
        for round_i in range(self.num_round):
            print(f"Round {round_i + 1}/{self.num_round} starting...")
            self.test_latest_model_on_testdata(round_i)
         
            local_model_paras_set = []
            if not self.data_queue.empty():
                client, client_weights, epoch = self.data_queue.get() 
                local_model_paras_set.append((client, client_weights, epoch))
                self.epoch = round_i + 1 # 全局迭代次数
                self.client_epoch = epoch
                self.latest_global_model = self.aggregate_async(client_weights)
                # self.test_latest_model_on_testdata(round_i)
                print("Data updated from Client ID: {}".format(
                                                                           client.id))
                client.response_queue.put((copy.deepcopy(self.latest_global_model), self.epoch)) 
                seq.append(client.id)
            else:
                time.sleep(1)

        print(seq)      
        self.test_latest_model_on_testdata(self.num_round)
        for client in self.clients:
            client.response_queue.put("STOP")
            self.running = False
        # # Save tracked information
        self.metrics.write()

    def local_train(self, clients):

        return 
   
    def aggregate_async(self, local_model_paras):
        # 检查这两个变量的类型
        with lock:  # 加锁，确保只有一个线程访问
            alpha = np.power(self.epoch - self.client_epoch + 1, -0.5)
            solution = torch.zeros_like(self.latest_global_model)
            solution = (1 - alpha) * self.latest_global_model + alpha * local_model_paras
        return solution

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = Async_CLient(self.options, i, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.clients_system_attr, self)
            all_client.append(local_client)

        return all_client