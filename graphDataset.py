import numpy as np
import torch
from temporal_motifs import motif
import datatable as dt
from datatable import dt, f, by, g, join, sort, update, ifelse
import collections
FType = torch.FloatTensor
LType = torch.LongTensor
import time
import random
class graphDataset(object):
    def __init__(self,neg_size,hist_len,win,graph,node_map_all,node_map,neg_node_map, neg_table_all,graph_idx_all,graph_idx2_all,graph_idx1_all,time_count_all):
        self.graph = graph
        self.neg_size, self.hist_len, self.win = neg_size,hist_len,win
        self.neg_table_size = int(1e5)
        self.NEG_SAMPLING_POWER = 0.75
        self.node_map_all = node_map_all
        self.node_map = node_map
        self.neg_node_map = neg_node_map
        self.neg_table = neg_table_all
        self.motif_p = motif(self.graph,self.win,graph_idx_all,graph_idx2_all,graph_idx1_all,time_count_all)

    def sample(self):
        start = time.perf_counter()
        s_nodes,t_node2,h_node2,n_node2,hist_times,h_t_time,t_times,h_t_masks,h_time_mask,type_all = self.extend_edges()
        sample = {
            's_nodes': s_nodes,
            't_node2': t_node2,
            'h_node2': h_node2,
            'n_node2': n_node2,
            'hist_times': hist_times,
            'h_t_time':h_t_time,
            't_times': t_times,
            'h_t_masks': h_t_masks,
            'h_time_mask':h_time_mask,
            'type_all':type_all,
        }
        end = time.perf_counter()
        timestap = end-start
        print('采样时间timestap: {:.0f}m {:.0f}s'.format(timestap // 60,timestap % 60))
        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, self.neg_size)
        sampled_nodes = []
        for j in self.neg_table[rand_idx]:
            sampled_nodes.append(self.node_map[int(j)])
        return np.array(sampled_nodes)

    def extend_edges(self):
        hist_times = []
        type_all = []
        h_t_time = []
        h_t_masks = []
        h_l_masks = []
        t_times = []
        hist_nodes_all = []
        h_time_mask = []
        neg_nodes = []
        s_nodes = []
        t_nodes = []
        
        dim = self.graph[:, dt.count()][0,0]
        for index in range(dim):
            row = self.graph[index,:]
            t_time = row[0,2]
            s_node = row[0,0]
            t_node = row[0,1]
            hist_nodes,hist_time,type_al,h_t_tim,h_t_mask,t_nod,h_l_mask = self.motif_p.type(row,index,self.hist_len)
            hist_nodes2 = []
            for i in range(len(hist_nodes)):
                hist_nodes2.append(self.node_map_all[hist_nodes[i]])

            hist_nodes_all.append(hist_nodes2)
            hist_times.append(hist_time)
            type_all.append(type_al)
            h_t_time.append(h_t_tim)
            h_t_masks.append(h_t_mask)
            h_l_masks.append(h_l_mask)
            t_times.append(int(t_time))

            neg_node = list(self.negative_sampling())
            neg_node2 = []
            for i in range(len(neg_node)):
                neg_node2.append(self.node_map_all[neg_node[i]])

            neg_nodes.append(neg_node2)
            s_nodes.append(self.node_map_all[s_node])
            t_nodes.append(self.node_map_all[t_node])
            
        edges_counts = dim
        np_hist_nodes_all = np.zeros((edges_counts,self.hist_len))
        for i in range(len(hist_nodes_all)):
            np_hist_nodes_all[i,:len(hist_nodes_all[i])] = hist_nodes_all[i]
        np_hist_nodes_all = torch.Tensor(np_hist_nodes_all)

        np_h_times = np.zeros((edges_counts,self.hist_len))
        for i in range(len(hist_times)):
            np_h_times[i,:len(hist_times[i])] = hist_times[i]
        np_h_times = torch.Tensor(np_h_times)

        np_t_times = np.zeros((edges_counts,self.hist_len,))
        for i in range(len(h_t_time)):
            np_t_times[i,:len(h_t_time[i])] = h_t_time[i]
        np_t_times = torch.Tensor(np_t_times)

        np_h_masks = np.zeros((edges_counts,self.hist_len,))

        for i in range(len(h_l_masks)):                       #全图的节点个数
            np_h_masks[i,:len(h_l_masks[i])] = h_l_masks[i]
        np_h_masks = torch.Tensor(np_h_masks)

        np_t_masks = np.zeros((edges_counts,self.hist_len,))
        for i in range(len(h_t_masks)):
            np_t_masks[i,:len(h_t_masks[i])] = h_t_masks[i]
        np_t_masks = torch.Tensor(np_t_masks)

        np_type_all = np.zeros((edges_counts,self.hist_len,))
        for i in range(len(type_all)):
            np_type_all[i,:len(type_all[i])] = type_all[i]
        np_type_all = torch.Tensor(np_type_all)

        n_node2 = neg_nodes

        s_nodes = torch.Tensor(s_nodes)
        t_nodes = torch.Tensor(t_nodes)
        n_node2 = torch.Tensor(n_node2)
        t_times = torch.Tensor(t_times)

        return s_nodes,t_nodes,np_hist_nodes_all,n_node2,np_h_times,np_t_times,t_times,np_t_masks,np_h_masks,np_type_all