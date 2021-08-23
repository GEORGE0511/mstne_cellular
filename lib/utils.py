import numpy as np
from datatable import dt, f, by, min, max
import collections
class utils(object):
    def __init__(self,history_length,num_for_predict):
        self.history_length = history_length
        self.num_for_predict = num_for_predict
        self.neg_table_size = int(1e5)
        self.NEG_SAMPLING_POWER = 0.75

    def read_and_generate_dataset(self,graph_signal_matrix_filename,graph_file):
        data_seq = np.load(graph_signal_matrix_filename)['arr_0']  # wd: (16992, 307, 3)
        data_seq = data_seq
        data_seq = np.float32(data_seq)  # wd: to reduce computation
        self.graph = dt.fread(graph_file) 
        return data_seq, self.graph

    def slice_data(self,data,train_mode,train_rate=0.6,points_per_hour=12):
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = 0
            end_index = data.shape[0] * train_rate - (self.history_length * points_per_hour + self.num_for_predict * points_per_hour)
        elif train_mode == "test":
            start_index = data.shape[0] * train_rate
            end_index = data.shape[0] - (self.history_length * points_per_hour + self.num_for_predict * points_per_hour)
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x_idx = []
        data_y_idx = []
        graph_all = []
        graph_idx_all = []
        graph_idx2_all = []
        graph_idx1_all = []
        node_map_all = []
        neg_node_map_all = []
        neg_table_all = []
        time_count_all = []

        for i in range(int((end_index - start_index) / (self.num_for_predict * points_per_hour))):
            print(i)
            temp = self.num_for_predict * points_per_hour

            a = i * self.num_for_predict * points_per_hour
            b = i * self.num_for_predict * points_per_hour + self.history_length * points_per_hour
            graph = self.graph[dt.rowall((f[2] >= a) & (f[2] <= b)), :]
            node_set = list(set(dt.unique(graph['C0']).to_list()[0]) | set(dt.unique(graph['C1']).to_list()[0]))
            node_dim = len(node_set)

            neg_node_map = {node_set[i]: i for i, k in enumerate(node_set)}
            node_map = {i: node_set[i] for i, k in enumerate(node_set)}

            graph[:,2] = graph[:, (f[2] - min(f[2]))/(max(f[2]) - min(f[2]))]

            degrees_1 = collections.Counter(dict(self.graph[:, {"counts": dt.count()}, by("C0")].to_tuples()))
            degrees_2 = collections.Counter(dict(self.graph[:, {"counts": dt.count()}, by("C1")].to_tuples()))
            degrees = dict(degrees_1 + degrees_2)

            if len(node_set) != 0:
                neg_table = self.init_neg_table(degrees,node_dim,node_map)
            
            graph_idx = {}
            graph_idx1 = {}
            graph_idx2 = {}
            time_count = {}
            for i in range(graph[:, dt.count()][0,0]):
                if str(graph[i,0]) + '_' + str(graph[i,0]) not in graph_idx:
                    graph_idx[str(graph[i,0]) + '_' + str(graph[i,1])] = []
                graph_idx[str(graph[i,0]) + '_' + str(graph[i,1])].append(i)

                if graph[i,0] not in graph_idx2:
                    graph_idx2[graph[i,0]] = []
                graph_idx2[graph[i,0]].append(i)

                if graph[i,1] not in graph_idx2:
                    graph_idx2[graph[i,1]] = []
                graph_idx2[graph[i,1]].append(i)

                graph_idx1[i] = [len(graph_idx2[graph[i,0]])-1,len(graph_idx2[graph[i,1]])-1]

                if (str(graph[i,1]) + '_' + str(graph[i,0])) in graph_idx:
                    for j in graph_idx[str(graph[i,1]) + '_' + str(graph[i,0])]:
                        if graph[i,2] == graph[j,2]:
                            time_count[i] = 2
                            time_count[j] = 2

                data_x_idx.append([a, b])
                data_y_idx.append([b, b + self.num_for_predict * points_per_hour])

                graph_all.append(graph)
                node_map_all.append(node_map)
                neg_node_map_all.append(neg_node_map)
                neg_table_all.append(neg_table)
                graph_idx_all.append(graph_idx)
                graph_idx2_all.append(graph_idx2)
                graph_idx1_all.append(graph_idx1)
                time_count_all.append(time_count)
            
        return data_x_idx, data_y_idx , graph_all, node_map_all, neg_node_map_all, neg_table_all, graph_idx_all, graph_idx2_all, graph_idx1_all, time_count_all
    
    def init_neg_table(self,degrees,node_dim,node_map):
        neg_table = np.zeros((self.neg_table_size,))
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(node_dim):
            tot_sum += np.power(degrees[node_map[k]], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(degrees[node_map[n_id]], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            neg_table[k] = n_id - 1
        return neg_table
    