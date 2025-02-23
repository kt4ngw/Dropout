from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import copy
class FedAvgTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        # self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(FedAvgTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model)
    
    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))
        #print("第一轮的模型", self.latest_global_model['fc2.bias'])
        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)
            
            selected_clients = self.select_clients()
            # Solve minimization locally
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)

            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)

        self.test_latest_model_on_testdata(self.num_round)

        self.metrics.write()

    def select_clients(self):
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        return select_clients

