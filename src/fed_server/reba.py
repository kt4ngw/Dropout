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
from src.fed_client.reba_client import Reba_CLient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Reba(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        # self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(Reba, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model)
        self.global_proto = None
 
    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))

        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)

            selected_clients = self.select_clients()

            local_model_paras_set, stats, local_model_proto_set = self.local_train(round_i, selected_clients, self.global_proto)  

            # 聚合参数
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)

            self.global_proto = self.aggregate_protos(local_model_proto_set)

            # self.optimizer.soft_decay_learning_rate()


        self.test_latest_model_on_testdata(self.num_round)

        # # Save tracked information
        self.metrics.write()

    def select_clients(self):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
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
        self.simple_average = False
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            num = 0
            for num_sample, local_solution in solns:
                # print(local_solution)
                num += num_sample
                averaged_solution += num_sample * local_solution
            averaged_solution /= num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()
    
    def aggregate_protos(self, protos_set, **kwargs):
        agg_protos_label = {}
        agg_sizes_label = {}    
        for local_sizes, local_protos in protos_set:
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                    agg_sizes_label[label].append(local_sizes[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
                    agg_sizes_label[label] = [local_sizes[label]]
        for [label, protos_list] in agg_protos_label.items():
            sizes_list = agg_sizes_label[label]
            proto = 0 * protos_list[0]
            for i in range(len(protos_list)):
                proto += sizes_list[i] * protos_list[i]
            agg_protos_label[label] = proto / sum(sizes_list)
        return agg_protos_label

    def local_train(self, round_i, select_clients, global_proto):
        local_model_paras_set = []
        stats = []
        local_model_proto_set = []
        for i, client in enumerate(select_clients, start=1):
            client.set_flat_model_params(self.latest_global_model)
            local_model_paras, stat, local_model_proto = client.local_train(round_i, global_proto)
            local_model_proto_set.append(local_model_proto)
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, client.id, i, int(self.per_round_c_fraction * self.clients_num),
                       stat['loss'], stat['acc']*100, stat['time'], ))
        return local_model_paras_set, stats, local_model_proto_set
    

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = Reba_CLient(self.options, i, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.clients_system_attr)
            all_client.append(local_client)

        return all_client