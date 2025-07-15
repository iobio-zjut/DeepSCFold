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


class esm_ss_predict_tri_step2(nn.Module):
    """
    Predicts contact maps as sigmoid(z_i W W W z_j + b)
    """

    def __init__(self, embed_dim=128, dim_1d=52, num_classes=1):
        super(esm_ss_predict_tri_step2, self).__init__()

        self.esm_embed_transform = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim))
        self.all_dim = embed_dim + dim_1d
        self.convnext_base = convnext_base(num_classes, self.all_dim)
        self.retnet_model = RetNet(layers=8, hidden_dim=180, ffn_size=360, heads=4)

    def forward(self, x0, x1):
        x0_2d = x0.unsqueeze(2).permute(0, 3, 1, 2)
        x1_2d = x1.unsqueeze(1).permute(0, 3, 1, 2)
        x_2d = torch.matmul(x0_2d, x1_2d)
        s = self.convnext_base(x_2d)
        ss = torch.sigmoid(s)

        return ss


class esm_ss_dataset_step2:
    def __init__(self, Pklfile):
        with open(Pklfile, 'rb') as handle:
            Pkl_dic = pickle.load(handle)

        self.x0_seqid = Pkl_dic['x0_seqid']
        self.x1_seqid = Pkl_dic['x1_seqid']
        self.x0_feature = Pkl_dic['x0_feature']
        self.x1_feature = Pkl_dic['x1_feature']

        print('# loaded', len(self.x0_feature), 'sequence pairs', file=sys.stderr)

    def __len__(self):
        return len(self.x0_feature)

    def __getitem__(self, i):
        return self.x0_seqid[i], self.x1_seqid[i], self.x0_feature[i], self.x1_feature[i]
