import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
from tqdm import tqdm
from util.util import get_pid_list, get_index_protein_dic
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from util.model_convnext import convnext_base
from util.Retnet.retnet_fn2 import RetNet


class esm_ss_predict_tri_step1(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W W W z_j + b)
    """

    def __init__(self, embed_dim=128, dim_1d=52, num_classes=1):
        super(esm_ss_predict_tri_step1, self).__init__()

        self.esm_embed_transform = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim))
        self.all_dim = embed_dim + dim_1d
        self.convnext_base = convnext_base(num_classes, self.all_dim)
        self.retnet_model = RetNet(layers=8, hidden_dim=180, ffn_size=360, heads=4)

    def forward(self, x0_esm, x0_1d):
        x0_esm = self.esm_embed_transform(x0_esm)
        x0 = torch.cat([x0_esm, x0_1d], dim=2)
        x0 = self.retnet_model(x0)

        return x0


class esm_ss_dataset_step1:
    def __init__(self, Pklfile):
        with open(Pklfile, 'rb') as handle:
            Pkl_dic = pickle.load(handle)
        # self.name = Pklfile
        self.x_seqid = Pkl_dic['x_seqid']
        self.x_esm = Pkl_dic['x_esm']
        self.x_1d = Pkl_dic['x_1d']
        print(self.x_seqid)
        print(self.x_esm)
        print(self.x_1d)

        print('# loaded', len(self.x_esm), 'sequence pairs', file=sys.stderr)

    def __len__(self):
        return len(self.x_esm)

    def __getitem__(self, i):
        # print(self.x_1d)
        min_length = 64
        dim_esm = 1280
        dim_1d = 52
        if len(self.x_esm[i]) < min_length:
            print(type(self.x_1d))
            print(self.x_1d.type)
            try:
                print(type(self.x_1d[i]))
                self.x_1d[i] = torch.tensor(self.x_1d[i])
                # print(type(self.x_1d[i]))
                self.x_esm[i] = torch.cat([self.x_esm[i], torch.zeros(min_length - self.x_esm[i].size(0), dim_esm)])
                self.x_1d[i] = torch.cat([self.x_1d[i], torch.zeros(min_length - self.x_1d[i].size(0), dim_1d)])
            except:
                print(self.x_seqid[i])
        return self.x_seqid[i], self.x_esm[i], self.x_1d[i]
