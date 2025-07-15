from __future__ import print_function, division
import sys
import os
import re

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

from os.path import isfile, isdir, join
import numpy as np
import gc
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import subprocess
import pickle
from scipy.stats import pearsonr, spearmanr
from util.model_retnet_conv1_step1 import esm_ss_dataset_step1, esm_ss_predict_tri_step1
from util.model_retnet_conv1_step2 import esm_ss_dataset_step2, esm_ss_predict_tri_step2
from util.util import make_parent_dir
from torch.utils.data import Dataset, DataLoader
from dataProcessingUtils import *

map = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


def cleanup():
    temp_files = [
        "query.pkl", "target.pkl", "features.pkl",
        "predict_pkl_dict.pkl", "features_step2.pkl"
    ]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    print("Cleanup complete.")


def extract_AAs_properties_ver1(aas):
    """Function to extract AA property, a part of 1D feature map.

    Args:
        param1 (aas): Amino acid sequence

    Returns:
        output: 2D marix of size feature by protein length.
        They contain 1-hot encoded residue, blosum column, normalized distance from terminus, meiler features.
    """

    _prop = np.zeros((20 + 24 + 1 + 7, len(aas)))
    for i in range(len(aas)):
        aa = aas[i]
        if aas[i] != 'OTH':
            _prop[residuemap[aa], i] = 1
            _prop[20:44, i] = blosummap[aanamemap[aa]]
            _prop[44, i] = min(i, len(aas) - i) * 1.0 / len(aas) * 2
            _prop[45:, i] = meiler_features[aa] / 5
        else:
            _prop[:20, i] = 0
            _prop[20:44, i] = 0
            _prop[44, i] = 0
            _prop[45:, i] = 0
    _prop = _prop.T
    return _prop


def get_fasta_dict(txt):
    fasta_dict = {}

    with open(txt, 'r') as fasta_file:
        current_key = None
        current_sequence = []

        for line in fasta_file:
            line = line.strip()

            if line.startswith('>'):
                if current_key is not None:
                    fasta_dict[current_key] = ''.join(current_sequence)

                current_key = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_key is not None:
            fasta_dict[current_key] = ''.join(current_sequence)
    return fasta_dict


def transform(sequence):
    seq = []
    for char in sequence:
        if char in map:
            seq.append(map[char])
        else:
            seq.append('OTH')
    return seq


def save_pairs_list_to_csv(filename, pairs_list):
    print("---------start save pairs---------")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Query', 'Target', 'Score'])
        for row in pairs_list:
            writer.writerow(row)


def getpickle(query_path, target_path, name):
    print("\n---------start get features---------")
    esm_2 = os.path.join(script_dir, "esm/saved_models/ESM-2/esm2_t33_650M_UR50D.pt")
    query_file = query_path
    query_esm2_out_path = f'{name}_query.pkl'
    embedding_generate_path = os.path.join(script_dir, "embedding_generate.py")
    command = f"python {embedding_generate_path} -emp {esm_2} -f {query_file} -e {query_esm2_out_path}"
    subprocess.run(command, shell=True, check=True)
    target_file = target_path
    target_esm2_out_path = f'{name}_target.pkl'
    command = f"python {embedding_generate_path} -emp {esm_2} -f {target_file} -e {target_esm2_out_path}"
    subprocess.run(command, shell=True, check=True)

    with open(query_esm2_out_path, 'rb') as handle1:
        query_esm2_dic = pickle.load(handle1)
    with open(target_esm2_out_path, 'rb') as handle2:
        target_esm2_dic = pickle.load(handle2)

    query_dict = get_fasta_dict(query_file)
    target_dict = get_fasta_dict(target_file)

    x_seqid = []
    x_esm = []
    x_1d = []
    x_dict = {}

    for query_key, query_value in query_esm2_dic.items():
        x_seqid.append(query_key)
        x_esm.append(query_value)
        x_1d.append(extract_AAs_properties_ver1(transform(query_dict[query_key])))

    for target_key, target_value in target_esm2_dic.items():
        x_seqid.append(target_key)
        x_esm.append(target_value)
        x_1d.append(extract_AAs_properties_ver1(transform(target_dict[target_key])))

    pairs_list = []
    for query_key, query_value in query_esm2_dic.items():
        for target_key, target_value in target_esm2_dic.items():
            pairs_list.append([query_key, target_key, 0.0])

    feature_dict = {
        'x_seqid': x_seqid,
        'x_esm': x_esm,
        'x_1d': x_1d
    }

    print("---------features complete---------\n")

    return feature_dict, pairs_list


def predict_step1(query_path, target_path, name):
    print("\n---------start predict TMscore---------")
    print("query: ", query_path)
    print("target: ", target_path)

    feature_dict, pairs_list = getpickle(query_path, target_path, name)

    feature_pkl = f'{name}_features.pkl'
    with open(feature_pkl, 'wb') as output_file:
        pickle.dump(feature_dict, output_file)

    pre_model_name = os.path.join(script_dir, "models/model_for_fast_rank.pkl")
    model_step1 = esm_ss_predict_tri_step1(embed_dim=128)
    checkpoint = torch.load(pre_model_name, map_location=lambda storage, loc: storage.cuda(0))
    model_step1.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model_step1 = torch.nn.DataParallel(model_step1)
    model_step1.to(device)
    ss_valid_dataset_step1 = esm_ss_dataset_step1(feature_pkl)
    print(type(ss_valid_dataset_step1))
    model_step1.eval()
    with torch.no_grad():
        predict_pkl_dict = {}

        for i in range(ss_valid_dataset_step1.__len__()):
            print(ss_valid_dataset_step1.__getitem__(i)[0])
            query_seqid = ss_valid_dataset_step1.__getitem__(i)[0]
            query_dict = ss_valid_dataset_step1.__getitem__(i)[1].to(device)
            query_dict_1d = torch.tensor(ss_valid_dataset_step1.__getitem__(i)[2], dtype=torch.float32).to(device)
            predict_pkl = model_step1(query_dict.unsqueeze(0), query_dict_1d.unsqueeze(0))
            predict_pkl_dict[query_seqid] = predict_pkl
    with open(f'{name}_predict_pkl_dict.pkl', 'wb') as output_file:
        pickle.dump(predict_pkl_dict, output_file)

    x0_seqid_data = []
    x0_feature_data = []
    x1_seqid_data = []
    x1_feature_data = []

    for pair in pairs_list:
        x0_seqid, x1_seqid, _ = pair
        x0_feature = predict_pkl_dict.get(x0_seqid, None)
        x1_feature = predict_pkl_dict.get(x1_seqid, None)

        if x0_feature is not None and x1_feature is not None:
            x0_seqid_data.append(x0_seqid)
            x0_feature_data.append(x0_feature)
            x1_seqid_data.append(x1_seqid)
            x1_feature_data.append(x1_feature)
        else:
            print(f"Error: Missing feature for pair ({x0_seqid}, {x1_seqid})")

    feature_data_dict = {
        'x0_seqid': x0_seqid_data,
        'x1_seqid': x1_seqid_data,
        'x0_feature': x0_feature_data,
        'x1_feature': x1_feature_data
    }
    feature_data_file = f'{name}_features_step2.pkl'

    with open(feature_data_file, 'wb') as output_file:
        pickle.dump(feature_data_dict, output_file)

    model_step2 = esm_ss_predict_tri_step2(embed_dim=128)
    checkpoint = torch.load(pre_model_name, map_location=lambda storage, loc: storage.cuda(0))
    model_step2.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model_step2 = torch.nn.DataParallel(model_step2)
    model_step2.to(device)

    ss_valid_dataset_step2 = esm_ss_dataset_step2(feature_data_file)

    model_step2.eval()

    score_list = []
    with torch.no_grad():
        for i in range(ss_valid_dataset_step2.__len__()):
            item = ss_valid_dataset_step2[i]
            query_seqid = item[0]
            target_seqid = item[1]
            query_feature = item[2].to(device)
            target_feature = item[3].to(device)

            predict_score = model_step2(query_feature, target_feature)
            score_list.append(predict_score.cpu().item())

    cleanup()

    print("---------return score_list---------\n")
    print(score_list)
    return score_list


if __name__ == '__main__':
    query_path = '/mydata/minghuah/MULTICOM/MSArank_model/examples/query1.fasta'
    target_path = '/mydata/minghuah/MULTICOM/MSArank_model/examples/target1.fasta'
    predict_step1(query_path, target_path, 'test')

