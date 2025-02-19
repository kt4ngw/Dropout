from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import copy
import torch
import torch.nn.functional as F


class MimicTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(MimicTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model, self.optimizer,)
        self.scores = np.zeros((self.options['num_of_clients'], self.options['num_of_clients']))
        self.number_simultaneous_participation = np.zeros((self.options['num_of_clients'], self.options['num_of_clients']))
        self.C = [torch.zeros_like(self.latest_global_model) for _ in range(self.options['num_of_clients'])]
    
    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))

        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)

            selected_clients = self.select_clients()

            if round_i == 0:
                selected_clients = self.select_clients(ALL=True)
            # print(selected_clients)
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)  
            # 修正模型参数
            new_local_model_paras_set = self.modified_model_para(local_model_paras_set, self.C)
            # 聚合参数
            self.latest_global_model = self.aggregate_parameters(new_local_model_paras_set)
            # 更新矫正变量
            # print("C前", self.C)
            self.updating_of_corrective_variables(self.latest_global_model, local_model_paras_set)
            # print("C后", self.C)
            self.optimizer.soft_decay_learning_rate()


        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        self.metrics.write()

    def select_clients(self, ALL=False):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        if ALL == True:
            num_clients = self.clients_num
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients
    

    
    def aggregate_parameters(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_global_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        self.simple_average = True
        if self.simple_average:
            num = 0
            for num_sample, _, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            num = 0
            for num_sample, _, local_solution in solns:
                # print(local_solution)
                num += num_sample
                averaged_solution += num_sample * local_solution
            averaged_solution /= num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()
    

    def modified_model_para(self, local_model_paras_set, C):

        # print("更新前", local_model_paras_set)

        for i in range(len(local_model_paras_set)):

            idx = local_model_paras_set[i][1]
            modified_tuple = list(local_model_paras_set[i]) 
            modified_tuple[2] += C[idx]  # 修改列表中的第三个元素
            local_model_paras_set[i] = tuple(modified_tuple) 

        # print("更新后", local_model_paras_set)

        return local_model_paras_set
    

    def updating_of_corrective_variables(self, latest_global_model, local_model_paras_set):

        # C = [torch.zeros_like(self.latest_global_model) for _ in range(self.options['num_of_clients'])]
        
        for i in range(len(local_model_paras_set)):
            idx = local_model_paras_set[i][1]
            self.C[idx] = latest_global_model - local_model_paras_set[i][2]

