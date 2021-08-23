
from lib.utils import utils
import torch
from torch.utils.data import Dataset
from graphDataset import graphDataset
from datatable import dt, f, by, min, max
class MSTNEDataset(Dataset):
    def __init__(self, data_path, graph_path, history_length, num_for_predict, points_per_hour, train_mode,neg_size,hist_len,win):
        """
        load processed data
        :param data_path: ["graph file name" , "flow data file name"], path to save the data file names
        :param num_nodes: number of nodes in graph
        :param divide_days: [ days of train data, days of test data], list to divide the original data
        :param time_interval: time interval between two traffic data records (mins)
        :param history_length: length of history data to be used
        :param train_mode: ["train", "test"]
        """
        self.data_path = data_path
        self.train_mode = train_mode
        self.history_length = history_length  # 6
        self.num_for_predict = num_for_predict # 1
        self.points_per_hour = points_per_hour  # 5 min = 12
        self.utils = utils(history_length,num_for_predict)
        self.data, self.graph = self.utils.read_and_generate_dataset(data_path,graph_path)

        self.data_x_idx, self.data_y_idx, self.graph_all, self.node_map, self.neg_node_map_all, \
            self.neg_table_all, self.graph_idx_all, self.graph_idx2_all, self.graph_idx1_all,\
            self.time_count_all  = self.utils.slice_data(self.data,train_mode,train_rate=0.6,points_per_hour=self.points_per_hour)
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.win = win
        self.node_set_all = list(set(dt.unique(self.graph['C0']).to_list()[0]) | set(dt.unique(self.graph['C1']).to_list()[0]))
        self.node_map_all = {self.node_set_all[i]: i for i, k in enumerate(self.node_set_all)}
        self.node_dim = len(self.node_set_all)

        if train_mode == 'train':
            self.data = self.data[:,0:self.node_dim]

    def __len__(self):
        return len(self.data_x_idx)

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        data_x = self.data[self.data_x_idx[index][0]:self.data_x_idx[index][1],:]
        data_y = self.data[self.data_y_idx[index][0]:self.data_y_idx[index][1],:]
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)
        graph = self.graph_all[index]
        node_map = self.node_map[index]
        neg_node_map = self.neg_node_map_all[index]
        neg_table_all = self.neg_table_all[index]
        graph_idx_all = self.graph_idx_all[index]
        graph_idx2_all = self.graph_idx2_all[index]
        graph_idx1_all = self.graph_idx1_all[index]
        time_count_all = self.time_count_all[index]
        samples = graphDataset(self.neg_size,self.hist_len,self.win,graph,self.node_map_all, node_map, \
            neg_node_map, neg_table_all,graph_idx_all,graph_idx2_all,graph_idx1_all,time_count_all)
        return {"graph": samples.sample(), "flow_x": data_x, "flow_y": data_y}