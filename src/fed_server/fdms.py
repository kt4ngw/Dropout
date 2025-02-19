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


class FDMSTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(FDMSTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model, self.optimizer,)
        self.scores = np.zeros((self.options['num_of_clients'], self.options['num_of_clients']))
        self.number_simultaneous_participation = np.zeros((self.options['num_of_clients'], self.options['num_of_clients']))
        # self.
    
    def train(self):
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))
        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)

            selected_clients = self.select_clients()

            local_model_paras_set, stats = self.local_train(round_i, selected_clients)
             
            # 更新得分~~~

            self.scores, self.number_simultaneous_participation = self.get_number_and_cosine_similarity(local_model_paras_set)

            # 生成补充参数~~~
            new_local_model_paras_set = self.get_supplementary_para(local_model_paras_set,  self.scores, ) 
            # print(new_local_model_paras_set)
            # 聚合参数~~~
            # 

            self.latest_global_model = self.aggregate_parameters(new_local_model_paras_set)

            self.optimizer.soft_decay_learning_rate()


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
    
    def get_number_and_cosine_similarity(self, local_model_paras_set):

        vectors = [(a, b) for _, a, b in local_model_paras_set]
        num_vectors = len(vectors)
        old_scores = copy.deepcopy(self.scores)
        new_scores = copy.deepcopy(self.scores)
        old_number_simultaneous_participation = copy.deepcopy(self.number_simultaneous_participation)
        for i in range(num_vectors - 1):
            for j in range(i, num_vectors):
                if vectors[i][0] != vectors[j][0]:   
                    cos_sim = F.cosine_similarity(vectors[i][1].unsqueeze(0), vectors[j][1].unsqueeze(0))
                    temp = old_number_simultaneous_participation[vectors[i][0], vectors[j][0]] / (old_number_simultaneous_participation[vectors[i][0], vectors[j][0]] + 1)
                    cos_value = temp  * old_scores[vectors[i][0], vectors[j][0]] + temp * cos_sim.item()
                    new_scores[vectors[i][0], vectors[j][0]] = cos_value
                    new_scores[vectors[j][0], vectors[i][0]] = cos_value  # 对称矩阵 

        for i in range(num_vectors - 1):
            for j in range(i, num_vectors):
                if vectors[i][0] != vectors[j][0]:
                    old_number_simultaneous_participation[vectors[i][0], vectors[j][0]] += 1
                    old_number_simultaneous_participation[vectors[j][0], vectors[i][0]] += 1  # 对称的关系，确保 (i, j) 和 (j, i) 都加 1

        return new_scores, old_number_simultaneous_participation



    def get_supplementary_para(self, local_model_paras_set, scores ):
        # print(local_model_paras_set)
        existing_ids = {item[1] for item in local_model_paras_set}
        all_ids = set(range(0, self.options['num_of_clients']))
        missing_ids = all_ids - existing_ids
        for missing_id in missing_ids:

            similarities = scores[missing_id] # 获取与 missing_id 相似的客户端的相似度向量
            valid_similarities = {i: similarities[i] for i in existing_ids}  # 排除 missing_id 自己
            max_similarity_idx = max(valid_similarities, key=valid_similarities.get)
            most_similar_client_params = next(
                params for _, client_id, params in local_model_paras_set if client_id == max_similarity_idx
            )
            
            # 创建一个空的 temp 张量，其形状与 latest_global_model 相同
            temp = copy.deepcopy(most_similar_client_params)  # 深拷贝最相似客户端的参数
            result = (self.clients_own_datavolume[missing_id], missing_id, temp)
            local_model_paras_set.append(result)
        # pass
        return local_model_paras_set
