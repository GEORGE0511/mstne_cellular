import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import sys
import torch.nn.functional as F
FType = torch.FloatTensor
LType = torch.LongTensor
import numpy as np
import time
import torch
import argparse
import configparser
import time
from dataloader import MSTNEDataset
import torch
import torch.nn
from torch.utils.data import DataLoader
# import torch.utils.tensorboard
# from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='configurations/MSTNE.conf',
                    help="configuration file path", required=False)
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_for_predict = int(data_config['num_for_predict'])
merge = bool(int(training_config['merge']))
points_per_hour = int(data_config['points_per_hour'])
batch_size = int(training_config['batch_size'])
neg_size = int(training_config['neg_size'])
hist_len = int(training_config['hist_len'])
win = int(training_config['win'])

class MSTNE():
    def __init__(self, emb_size=64, learning_rate=0.0001, batch_size=1, save_step=10, epoch_num=100):
        super(MSTNE, self).__init__()
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num
        self.win = win
        self.MSTNEDataset = MSTNEDataset('array_save.npz', \
            'user_move.txt', 12, 3, 4, "train", neg_size,hist_len,win)
        self.node_dim = self.MSTNEDataset.node_dim

        self.node_emb = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, self.emb_size))
                ).type(FType), requires_grad=True)
            
        self.delta = torch.nn.Parameter((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

        self.att_param = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
        -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim,24))).type(
        FType), requires_grad=True)

        self.att_param_p = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
        -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim,24))).type(
        FType), requires_grad=True)

        self.FCL = torch.nn.Sequential(
            torch.nn.Linear(48, 24),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(24, 24),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(24, 12),
            torch.nn.LeakyReLU(inplace=True)
        )

        self.opt = SGD(lr=learning_rate,params=[self.node_emb, self.att_param, self.att_param_p, self.delta] + list(self.FCL.parameters()))
        self.loss = torch.FloatTensor()
        self.edge_count = torch.FloatTensor()

    def forward(self,flow_x,flow_y,s_nodes,t_node2,h_node2,n_node2,hist_times,h_t_time,t_times,h_t_masks,h_time_mask,type_all):

        batch = s_nodes.size()[0] 
        s_node_emb2 = self.node_emb.index_select(0, (s_nodes.view(-1))).view(batch, -1)
        t_node_emb2 = self.node_emb.index_select(0, (t_node2.view(-1))).view(batch, -1)
 
        h_node_emb2 = self.node_emb.index_select(0, (h_node2.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, (n_node2.view(-1))).view(batch, self.neg_size, -1)

        h_types = torch.nn.functional.one_hot(type_all.to(torch.int64), num_classes=24)
        att_para = self.att_param.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        att_param_p = self.att_param_p.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        att = softmax((h_types * att_para.unsqueeze(1)).sum(dim=2), dim=1)
        att_p = softmax((h_types * att_param_p.unsqueeze(1)).sum(dim=2), dim=1)
    
        p_mu = ((s_node_emb2 - t_node_emb2)**2).sum(dim=1).neg()
        p_alpha = ((h_node_emb2 - t_node_emb2.unsqueeze(1))**2).sum(dim=2).neg()
        p_t_alpha = ((h_node_emb2 - s_node_emb2.unsqueeze(1))**2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - hist_times)  # (batch, hist_len)
        d_t_time = torch.abs(t_times.unsqueeze(1) - h_t_time)  # (batch, hist_len)

        a = p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_t_masks)
        b = p_t_alpha * torch.exp(delta * Variable(d_t_time)) * Variable(h_time_mask)
        p_lambda = p_mu + ((att * (a+b) - att_p * (abs(b-a))).sum(dim=1))

        n_mu = ((s_node_emb2.unsqueeze(1) - n_node_emb)**2).sum(dim=2).neg()
        n_alpha = ((h_node_emb2.unsqueeze(2) - n_node_emb.unsqueeze(1))**2).sum(dim=3).neg()
        n_t_alpha = ((h_node_emb2 - s_node_emb2.unsqueeze(1))**2).sum(dim=2).neg()
        a = n_alpha * torch.exp(delta * (Variable(d_time))).unsqueeze(2) * (Variable(h_t_masks).unsqueeze(2))
        b = n_t_alpha.unsqueeze(1) * torch.exp(delta * (Variable(d_t_time))).unsqueeze(2) * (Variable(h_time_mask).unsqueeze(2))
        n_lambda = n_mu + ((att.unsqueeze(2) * (a+b) - att_p.unsqueeze(2) * (abs(b-a))).sum(dim=1))

        z_i = F.normalize(self.node_emb, dim=1)
        z_j = F.normalize(self.node_emb, dim=1)
        similarity_matrix = (F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2) + 1) / 2
        similarity_matrix = similarity_matrix - torch.diag_embed(torch.diag(similarity_matrix))

        flow_x = torch.transpose(flow_x.squeeze(0),1,0) + torch.mm(similarity_matrix , torch.transpose(flow_x.squeeze(0),1,0))
        flow_y = torch.transpose(flow_y.squeeze(0),1,0)
        pre_y = self.FCL(flow_x)
        crirerion = torch.nn.MSELoss()
        pre_loss = crirerion(pre_y,flow_y)

        return p_lambda, n_lambda, pre_loss

    def loss_func(self,flow_x,flow_y,s_nodes,t_node2,h_node2,n_node2,hist_times,h_t_time,t_times,h_t_masks,h_time_mask,type_all):

        p_lambdas, n_lambdas, pre_loss = self.forward(flow_x,flow_y,s_nodes,t_node2,h_node2,n_node2,hist_times,h_t_time,t_times,h_t_masks,h_time_mask,type_all)
        loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
            torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)  + pre_loss

        return loss

    def update(self,flow_x,flow_y,s_nodes,t_node2,h_node2,n_node2,hist_times,h_t_time,t_times,h_t_masks,h_time_mask,type_all):
        self.opt.zero_grad()
        loss = self.loss_func(flow_x,flow_y,s_nodes,t_node2,h_node2,n_node2,hist_times,h_t_time,t_times,h_t_masks,h_time_mask,type_all)
        loss = loss.sum()
        self.loss += loss.data
        self.edge_count += s_nodes.shape[0]
        loss.backward()
        self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            self.edge_count = 0.0
            loader = DataLoader(self.MSTNEDataset, batch_size=self.batch, shuffle=True, num_workers=0)

            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings('./emb/dblp_htne_attn_%d.emb' % (epoch),self.node_emb.cpu().data.numpy())

            for i_batch, sample_batched in enumerate(loader):
                self.update(
                    sample_batched['flow_x'].type(FType),
                    sample_batched['flow_y'].type(FType),
                    sample_batched['graph']['s_nodes'].squeeze(0).type(LType),
                    sample_batched['graph']['t_node2'].squeeze(0).type(LType),
                    sample_batched['graph']['h_node2'].squeeze(0).type(LType),
                    sample_batched['graph']['n_node2'].squeeze(0).type(LType),
                    sample_batched['graph']['hist_times'].squeeze(0).type(FType),
                    sample_batched['graph']['h_t_time'].squeeze(0).type(FType),
                    sample_batched['graph']['t_times'].squeeze(0).type(FType),
                    sample_batched['graph']['h_t_masks'].squeeze(0).type(FType),
                    sample_batched['graph']['h_time_mask'].squeeze(0).type(FType),
                    sample_batched['graph']['type_all'].squeeze(0).type(LType),
                )
                
                sys.stdout.write("\repoch " + "{} / {}".format(epoch,self.epochs) + '\tbatch: ' + "{} / {}".format(i_batch,len(self.MSTNEDataset.data_x_idx)) + \
                    '\tloss: ' + str(self.loss.cpu().numpy() / self.edge_count) + '\tdelta:' + str(self.delta.mean().cpu().data.numpy()))
                sys.stdout.flush()

            # sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
            #                  str(self.loss.cpu().numpy() / 419) + '\n')
            # sys.stdout.flush()

                # timestap = time.time() - since
                # print('timestap: {:.0f}m {:.0f}s'.format(timestap // 60,timestap % 60))

        # self.save_node_embeddings('./emb/dblp_htne_attn_%d.emb' % (self.epochs))

    def evaluate(self,epoch):
        neigh_agg_p_all = self.node_emb.cpu().data.numpy()
        label = list(self.data.label_map.values())
        clf = LogisticRegression()
        scores = cross_val_score(clf, neigh_agg_p_all, label, cv=5, scoring='f1_weighted', n_jobs=8)
        print("val_scores:{}".format(scores))

        self.save_node_embeddings('./emb/dblp_htne_attn_%d.emb' % (epoch),neigh_agg_p_all)


    def save_node_embeddings(self, path,neigh_agg_p_all):
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in neigh_agg_p_all[n_idx]) + '\n')
        writer.close()

    # def collate_fn(self,data): 

    #     temp = data[0]
    #     flow_x = temp['flow_x'].type(FType)
    #     flow_y = temp['flow_y'].type(FType)
    #     s_nodes = temp['graph']['s_nodes'].squeeze(0).type(LType)
    #     t_node2 = temp['graph']['t_node2'].squeeze(0).type(LType)
    #     h_node2 = temp['graph']['h_node2'].squeeze(0).type(LType)
    #     n_node2 = temp['graph']['n_node2'].squeeze(0).type(LType)
    #     hist_times = temp['graph']['hist_times'].squeeze(0).type(FType)
    #     h_t_time = temp['graph']['h_t_time'].squeeze(0).type(FType)
    #     t_times = temp['graph']['t_times'].squeeze(0).type(FType)
    #     h_t_masks = temp['graph']['h_t_masks'].squeeze(0).type(FType)
    #     h_time_mask = temp['graph']['h_time_mask'].squeeze(0).type(FType)
    #     type_all = temp['graph']['type_all'].squeeze(0).type(LType)

    #     for each in data[1:]:
    #         flow_x = torch.cat((flow_x,each['flow_x'].type(FType)),dim=0)
    #         flow_y = torch.cat((flow_y,each['flow_y'].type(FType)),dim=0)
    #         s_nodes = torch.cat((s_nodes,each['graph']['s_nodes'].squeeze(0).type(LType)),dim=0)
    #         t_node2 = torch.cat((t_node2,each['graph']['t_node2'].squeeze(0).type(LType)),dim=0)
    #         h_node2 = torch.cat((h_node2,each['graph']['h_node2'].squeeze(0).type(LType)),dim=0)
    #         n_node2 = torch.cat((n_node2,each['graph']['n_node2'].squeeze(0).type(LType)),dim=0)
    #         hist_times = torch.cat((hist_times,each['graph']['hist_times'].squeeze(0).type(FType)),dim=0)
    #         h_t_time = torch.cat((h_t_time,each['graph']['h_t_time'].squeeze(0).type(FType)),dim=0)
    #         t_times = torch.cat((t_times,each['graph']['t_times'].squeeze(0).type(FType)),dim=0)
    #         h_t_masks = torch.cat((h_t_masks,each['graph']['h_t_masks'].squeeze(0).type(FType)),dim=0)
    #         h_time_mask = torch.cat((h_time_mask,each['graph']['h_time_mask'].squeeze(0).type(FType)),dim=0)
    #         type_all = torch.cat((type_all,each['graph']['type_all'].squeeze(0).type(LType)),dim=0)

    #     return {
    #         'flow_x':flow_x,
    #         'flow_y':flow_y,
    #         'graph':{
    #         's_nodes': s_nodes,
    #         't_node2': t_node2,
    #         'h_node2': h_node2,
    #         'n_node2': n_node2,
    #         'hist_times': hist_times,
    #         'h_t_time':h_t_time,
    #         't_times': t_times,
    #         'h_t_masks': h_t_masks,
    #         'h_time_mask':h_time_mask,
    #         'type_all':type_all,
    #         }
    #     }


if __name__ == '__main__':
    htne = MSTNE()
    htne.train()

 


