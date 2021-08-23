'''
Author: your name
Date: 2021-03-03 19:05:14
LastEditTime: 2021-03-23 14:24:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \mctne\motif.py
'''
import random

class motif(object):
    def __init__(self,all_edges_info,win,graph_idx_all,graph_idx2_all,graph_idx1_all,time_count_all):
        self.all_edges_info = all_edges_info
        self.win = win
        self.graph_idx_all = graph_idx_all
        self.graph_idx2_all = graph_idx2_all
        self.graph_idx1_all = graph_idx1_all
        self.time_count_all = time_count_all

    def find_type_MTNE(self,left_edges,right_edges,i,j,tp):
        if tp == 3:
            return 9
        if tp == 2:
            if left_edges[i][2] == -1:
                return 5
            else:
                return 6
        if tp == 1:
            if right_edges[j][2] == -1:
                return 7
            else:
                return 8
        if tp == 0:
            if left_edges[i][2] == -1 and right_edges[j][2] == -1:
                return 1
            elif left_edges[i][2] == 1 and right_edges[j][2] == -1:
                return 3
            elif left_edges[i][2] == -1 and right_edges[j][2] == 1:
                return 4
            else:
                return 2
    
    def find_type_MCTNE(self,left_time,right_time,l_type,r_type):
        if l_type == 1 and r_type == 1:
            tp = 2
        elif l_type == 0 and r_type == 0:
            tp = 3
        elif l_type == 0 and r_type == 1:
            tp = 4
        elif l_type == 1 and r_type == 0:
            tp = 5
        elif l_type == 1 and r_type == 2:
            tp = 6
        elif l_type == 0 and r_type == 2:
            tp = 7
        elif l_type == 2 and r_type == 0:
            tp = 8
        elif l_type == 2 and r_type == 1:
            tp = 9
        elif l_type == 2 and r_type == 2:
            tp = 10
        if left_time < right_time:
            return 2 * tp +1
        else:
            return 2 * tp

    def type(self,all_edges,index,hist_len):
        motif_edges = []
        t_nodes = []
        type_all = []
        hist_nodes = []
        hist_time = []
        h_t_time = []
        h_t_masks = [0 for i in range(hist_len)]
        h_l_masks = [0 for i in range(hist_len)]

        next_edges = all_edges
        s_node = next_edges[0,0]
        t_node = next_edges[0,1]
        time = next_edges[0,2]
        
        inx = 0
        while len(hist_nodes)  < hist_len:
            l_neighbors = self.graph_idx2_all[s_node][0:self.graph_idx1_all[index][0]]
            r_neighbors = self.graph_idx2_all[t_node][0:self.graph_idx1_all[index][1]]

            l_counts = len(l_neighbors)
            r_counts = len(r_neighbors)

            if l_counts + r_counts == 0:
                break

            index_2 = random.sample(list(range(l_counts + r_counts)),1)[0]

            l_type = -1
            if index_2 <= l_counts - 1:
                a_index = l_neighbors[index_2]
                if a_index in self.time_count_all:
                    l_type = 2
                l_s_node = self.all_edges_info[a_index,0]
                l_t_node = self.all_edges_info[a_index,1]
                l_time = self.all_edges_info[a_index,2]

            else:
                a_index = r_neighbors[index_2-l_counts]
                if a_index in self.time_count_all:
                    l_type = 2
                l_s_node = self.all_edges_info[a_index,0]
                l_t_node = self.all_edges_info[a_index,1]
                l_time = self.all_edges_info[a_index,2]
            
            one_type = -1
            if l_s_node == s_node:
                one_type = 0
                if l_type == -1:
                    l_type = 0
                if (str(l_t_node)+ '_' + str(t_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_t_node)+ '_' + str(t_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        r_time = self.all_edges_info[r_index,2]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        motif_edges.append([l_t_node,l_time,r_time,self.find_type_MCTNE(l_time,r_time,l_type,r_type)])

                elif (str(t_node)+ '_' + str(l_t_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(t_node)+ '_' + str(l_t_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0
                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_t_node,l_time,r_time,self.find_type_MCTNE(l_time,r_time,l_type,r_type)])

            elif l_t_node == s_node:
                one_type = 1
                if l_type == -1:
                    l_type = 1

                if (str(l_s_node)+ '_' + str(t_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_s_node)+ '_' + str(t_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_s_node,l_time,r_time,self.find_type_MCTNE(l_time,r_time,l_type,r_type)])

                if (str(t_node)+ '_' + str(l_s_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(t_node)+ '_' + str(l_s_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0

                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_s_node,l_time,r_time,self.find_type_MCTNE(l_time,r_time,l_type,r_type)])
            
            elif l_s_node == t_node:
                one_type = 2
                if l_type == -1:
                    l_type = 0
                if (str(l_t_node)+ '_' + str(s_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_t_node)+ '_' + str(s_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MCTNE(r_time,l_time,r_type,l_type)])
                
                if (str(s_node)+ '_' + str(l_t_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(s_node)+ '_' + str(l_t_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:

                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0
                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MCTNE(r_time,l_time,r_type,l_type)])
            
            else:
                one_type = 3
                if l_type == -1:
                    l_type = 1
                if (str(l_s_node)+ '_' + str(s_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_s_node)+ '_' + str(s_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MCTNE(r_time,l_time,r_type,l_type)])

                if (str(s_node)+ '_' + str(l_s_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(s_node)+ '_' + str(l_s_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0
                        r_time = self.all_edges_info[r_index,2]
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MCTNE(r_time,l_time,r_type,l_type)])

            if len(motif_edges) == 0:
                hist_nodes.append(int(l_s_node))
                hist_time.append(l_time)
                h_t_time.append(l_time)

                if l_type == 2 and one_type <= 1:
                    type_all.append(22)
                elif l_type == 2 and one_type > 1:
                    type_all.append(23)
                else:
                    type_all.append(one_type)
                
                if type_all == 2 or type_all == 3 or type_all == 24:
                    h_t_masks[inx] = 1
                    h_l_masks[inx] = 0
                    inx += 1
                else:
                    h_t_masks[inx] = 0
                    h_l_masks[inx] = 1
                    inx += 1

                if index_2 < l_counts :
                    next_edges = self.all_edges_info[l_neighbors[index_2],:]
                else:
                    next_edges = self.all_edges_info[r_neighbors[index_2-l_counts],:]

                s_node = next_edges[0,0]
                t_node = next_edges[0,1]
                time = next_edges[0,2]
                t_nodes.append(int(t_node))
            
            else:
                select_edges = random.randint(0,len(motif_edges)-1)
                hist_nodes.append(int(motif_edges[select_edges][0]))
                hist_time.append(motif_edges[select_edges][1])
                h_t_time.append(motif_edges[select_edges][2])

                h_t_masks[inx] = 1
                h_l_masks[inx] = 1
                inx += 1

                type_all.append(motif_edges[select_edges][3])
                if motif_edges[select_edges][1] < motif_edges[select_edges][2]:
                    s_node = s_node
                    t_node = int(motif_edges[select_edges][0])
                    time = motif_edges[select_edges][1]
                else:
                    s_node = int(motif_edges[select_edges][0])
                    t_node = t_node
                    time = motif_edges[select_edges][2]

        return hist_nodes,hist_time,type_all,h_t_time,h_t_masks,t_nodes,h_l_masks