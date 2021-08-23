from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import graphscope.learning.graphlearn as gl
import numpy as np
from deepwalk import DeepWalk
config = {'walk_len': 20,
            'window_size': 5,
            'node_count': 10312,
            'hidden_dim': 128,
            'batch_size': 128,
            'neg_num': 10,
            'epoch': 40,
            'learning_algo': 'adam',
            'learning_rate': 0.01,
            'emb_save_dir': "./id_emb",
            's2h': False,
            'ps_hosts': None,
            'temperature': 1.0,
            'node_type': 'item',
            'edge_type': 'relation'}

node_type = config['node_type']
edge_type = config['edge_type']
g = gl.Graph().edge("/home/qiaozhi/MSTNE/graphscope/user_move.txt",
                    edge_type=(node_type, node_type, edge_type),
                    decoder=gl.Decoder(weighted=True), directed=False)\
            .node("/home/qiaozhi/MSTNE/graphscope/ap.txt", node_type=node_type,
                    decoder=gl.Decoder(weighted=True))
TRACKER_PATH  = './tracker/'
os.system('mkdir -p %s' % TRACKER_PATH)
os.system('rm -rf %s*' % TRACKER_PATH)
g.init(server_id=0, server_count=1, tracker=TRACKER_PATH)

def train(config, graph):
    deepwalk = DeepWalk(graph,
                    config['walk_len'],
                    config['window_size'],
                    config['node_count'],
                    config['hidden_dim'],
                    config['neg_num'],
                    config['batch_size'],
                    s2h=config['s2h'],
                    ps_hosts=config['ps_hosts'],
                    temperature=config['temperature'])
    for i in range(10):
        sub_edges =  deepwalk._sample_seed()
        neighbors = deepwalk._positive_sample(sub_edges.emit())

train(config, g)