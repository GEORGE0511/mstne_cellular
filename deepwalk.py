# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""class of DeepWalk model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import graphscope.learning.graphlearn as gl
try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import numpy as np
import random

class DeepWalk(gl.LearningBasedModel):
  """
  Args:
    graph: Initialized gl.Graph object.
    walk_len: Random walk length.
    window_size: Window size.
    node_count: Total numebr of nodes.
    batch_size: Batch size for training set.
    hidden_dim: Hidden dimension.
    neg_num: The number of negative samples for each node.
    s2h: Set it to True if using string2hash.
    ps_hosts: Set when running in distributed mode.
    temperature: Softmax temperature.
    node_type: User defined node type name.
    edge_type: User defined edge type name.
  """
  def __init__(self,
               graph,
               walk_len,
               window_size,
               node_count,
               hidden_dim,
               neg_num,
               batch_size,
               s2h=False,
               ps_hosts=None,
               temperature=1.0,
               node_type='item',
               edge_type='relation'):
    super(DeepWalk, self).__init__(graph,
                                   batch_size)
    self.walk_len = walk_len
    self.window_size = window_size
    self.node_count = node_count
    self.hidden_dim = hidden_dim
    self.neg_num = neg_num
    self.s2h = s2h
    self.ps_hosts=ps_hosts
    self.temperature=temperature
    self.node_type = node_type
    self.edge_type = edge_type

    # construct EgoSpecs.
    self.ego_spec = gl.EgoSpec(gl.FeatureSpec(0, 0))
    # encoders.
    self.encoders = self._encoders()

  def _sample_seed(self, mode='train'):
    return self.graph.E(self.edge_type).batch(5)

  def _positive_sample(self, t):
    
    src_out_edges = self.graph.E(edge_type="relation", feed=t).outV().outE("relation").sample(10).by("random").emit()
    src_in_edges = self.graph.E(edge_type="relation", feed=t).outV().inE("relation").sample(10).by("random").emit()
    dst_out_edges = self.graph.E(edge_type="relation", feed=t).inV().outE("relation").sample(10).by("random").emit()
    dst_in_edges = self.graph.E(edge_type="relation", feed=t).inV().inE("relation").sample(10).by("random").emit()
    all_edges = [src_out_edges[2],src_in_edges[2], dst_out_edges[2], dst_in_edges[2]]
    
    src_ids = src_out_edges[2].src_ids + src_in_edges[2].src_ids + dst_out_edges[2].src_ids + dst_in_edges[2].src_ids
    dst_ids = src_out_edges[2].dst_ids + src_in_edges[2].dst_ids + dst_out_edges[2].dst_ids + dst_in_edges[2].dst_ids
    
    print(src_ids)
    print(dst_ids)


    
    a = len(dst_ids)
    neighbors = list()
    while len(neighbors) < self.neg_num: 
      hl_idx = random.sample(range(0,a),1)
      first_src_node = src_ids[hl_idx]
      first_dst_node = dst_ids[hl_idx]
      if first_src_node != t.dst_ids and first_src_node != t.dst_ids:
        temp_node = first_src_node
        temp_node2 = first_dst_node
        first_edge_type = 1
      else:
        temp_node = first_dst_node
        temp_node2 = first_src_node
        first_edge_type = 0
      temporal_motifs = []
      for i in range(a):
        if i != hl_idx:
          src_node_temp = src_ids[i]
          dst_node_temp = dst_ids[i]
          if temp_node2 == t.dst_ids:
            if temp_node == src_node_temp and dst_node_temp == t.dst_ids:
              temporal_motif_type =  self.find_type_MCTNE()
              temporal_motifs.append([temp_node,time1,time2,type])
            elif temp_node == dst_node_temp and src_node_temp ==  t.dst_ids:
              temporal_motif_type =  self.find_type_MCTNE()
              temporal_motifs.append([temp_node,time1,time2,type])
      if len(temporal_motifs) == 0:
        neighbors.append(first_src_node,first_dst_node,time,0,type)
      else:
        temp_idx = random.sample(range(0,len(temporal_motifs)))
        motif = temporal_motifs[temp_idx]
        neighbors.append()


  def find_type_MCTNE(self,next_edges,i,j,tp):
    left_edges = self.all_edges_info.iloc[i]
    right_edges = self.all_edges_info.iloc[j]
    if tp == 3:
        if left_edges['Time'] >= right_edges['Time']:
            return 17
        else:
            return 18
    if tp == 2:
        if next_edges['start_ap'] == left_edges['end_ap']:
            if left_edges['Time'] >= right_edges['Time']:
                return 9
            else:
                return 10
        else:
            if left_edges['Time'] >= right_edges['Time']:
                return 11
            else:
                return 12
    if tp == 1:
        if next_edges['end_ap'] == right_edges['end_ap']:
            if left_edges['Time'] >= right_edges['Time']:
                return 13
            else:
                return 14
        else:
            if left_edges['Time'] >= right_edges['Time']:
                return 15
            else:
                return 16
    if tp == 0:
        if next_edges['start_ap'] == left_edges['end_ap'] and next_edges['end_ap'] == right_edges['end_ap']:
            if left_edges['Time'] >= right_edges['Time']:
                return 1
            else:
                return 2
        if next_edges['start_ap'] == left_edges['start_ap'] and next_edges['end_ap'] == right_edges['end_ap']:
            if left_edges['Time'] >= right_edges['Time']:
                return 5
            else:
                return 6
        if next_edges['start_ap'] == left_edges['end_ap'] and next_edges['end_ap'] == right_edges['start_ap']:
            if left_edges['Time'] >= right_edges['Time']:
                return 7
            else:
                return 8
        else:
            if left_edges['Time'] >= right_edges['Time']:
                return 3
            else:
                return 4





    # for i in 
    # src_out_edges_times = src_out_edges[2].float_attrs
    # src_out_edges_nodes = src_out_edges[2].src_ids()
    # src_in_edges_times = src_in_edges[2].float_attrs
    # dst_out_edges_times = dst_out_edges[2].float_attrs
    # dst_in_edges_times = dst_in_edges[2].float_attrs
