from __future__ import print_function, division

import sys
import os
import re
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

from os.path import isfile, isdir, join
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import subprocess
import pickle
import csv
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


def extract_AAs_properties_ver1(aas):
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

def process_msa_file(msa_file_path, max_sequences=5120):
    output_file_path = None

    with open(msa_file_path, 'r') as msa_file:
        lines = msa_file.readlines()

    sequences = []
    current_sequence = []

    for line in lines:
        if line.startswith('>'):
            if current_sequence:
                sequences.append(''.join(current_sequence))
                current_sequence = []

            sequences.append(line.strip())
        else:
            current_sequence.append(line.strip())

    if current_sequence:
        sequences.append(''.join(current_sequence))

    sequence_count = len([seq for seq in sequences if seq.startswith('>')])

    if sequence_count > max_sequences:
        truncated_sequences = []
        sequence_counter = 0

        for item in sequences:
            if item.startswith('>'):
                if sequence_counter >= max_sequences:
                    break
                sequence_counter += 1

            truncated_sequences.append(item)

        output_file_path = f"{msa_file_path}_{max_sequences}"

        with open(output_file_path, 'w') as output_file:
            output_file.write('\n'.join(truncated_sequences) + '\n')

        return output_file_path

    return msa_file_path

def get_fasta_dict(txt, max_sequences=None):
    fasta_dict = {}
    with open(txt, 'r') as fasta_file:
        current_key = None
        current_sequence = []
        sequence_count = 0

        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                if current_key is not None:
                    fasta_dict[current_key] = ''.join(current_sequence)
                    sequence_count += 1
                    if max_sequences is not None and sequence_count >= max_sequences:
                        break

                current_key = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_key is not None and (max_sequences is None or sequence_count < max_sequences):
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


def load_pairs_from_csv(filename):
    pairs_list = []
    with open(filename, 'r') as csvfile:
        for line in csvfile:
            row = re.split(r'[ \t,]+', line.strip())
            pairs_list.append([row[0], row[1], 0])
    return pairs_list


def getpickle(query_path, target_path, output_dir, esm_path, embedding_script):
    print("---------start get features---------")
    query_file = query_path
    query_esm2_out_path = os.path.join(output_dir, 'query.pkl')
    command = f"python {embedding_script} -emp {esm_path} -f {query_file} -e {query_esm2_out_path}"
    subprocess.run(command, shell=True, check=True)

    query_dict = get_fasta_dict(query_file)

    with open(query_esm2_out_path, 'rb') as handle1:
        query_esm2_dic = pickle.load(handle1)

    x_seqid = []
    x_esm = []
    x_1d = []

    for query_key, query_value in query_esm2_dic.items():
        x_seqid.append(query_key)
        x_esm.append(query_value)
        x_1d.append(extract_AAs_properties_ver1(transform(query_dict[query_key])))
        # print(len(x_1d))

    if target_path:
        target_file = target_path
        target_esm2_out_path = os.path.join(output_dir, 'target.pkl')
        command = f"python {embedding_script} -emp {esm_path} -f {target_file} -e {target_esm2_out_path}"
        subprocess.run(command, shell=True, check=True)

        target_dict = get_fasta_dict(target_file)

        with open(target_esm2_out_path, 'rb') as handle2:
            target_esm2_dic = pickle.load(handle2)

        for target_key, target_value in target_esm2_dic.items():
            x_seqid.append(target_key)
            x_esm.append(target_value)
            x_1d.append(extract_AAs_properties_ver1(transform(target_dict[target_key])))

    data_dict = {
        'x_seqid': x_seqid,
        'x_esm': x_esm,
        'x_1d': x_1d
    }

    features_out_path = os.path.join(output_dir, 'features.pkl')
    with open(features_out_path, 'wb') as output_file:
        pickle.dump(data_dict, output_file)

    return features_out_path


def predict_step1(query_path, target_path, pairs_file, output_dir, esm_path, embedding_script):
    print("---------start predict TMscore---------")
    print("query: ", query_path)
    if pairs_file:
        print("pairs_file: ", pairs_file)
        pairs_list = load_pairs_from_csv(pairs_file)
    else:
        print("target: ", target_path)
        query_dict = get_fasta_dict(args.query_path)
        target_dict = get_fasta_dict(args.target_path)
        pairs_list = [[query_key, target_key, 0.0] for query_key in query_dict.keys() for target_key in
                      target_dict.keys()]

    predict_pkl = getpickle(query_path, target_path, output_dir, esm_path, embedding_script)
    pre_model_name = os.path.join(script_dir, "models/model_for_fast_rank.pkl")

    model_step1 = esm_ss_predict_tri_step1(embed_dim=128)
    checkpoint = torch.load(pre_model_name, map_location=lambda storage, loc: storage.cuda(0))
    model_step1.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_step1.to(device)

    ss_valid_dataset = esm_ss_dataset_step1(predict_pkl)
    print(ss_valid_dataset.__getitem__(0))
    print(ss_valid_dataset)

    model_step1.eval()
    with torch.no_grad():
        predict_pkl_dict = {}
        for i in range(ss_valid_dataset.__len__()):
            query_seqid = ss_valid_dataset.__getitem__(i)[0]
            query_dict = ss_valid_dataset.__getitem__(i)[1].to(device)
            query_dict_1d = torch.tensor(ss_valid_dataset.__getitem__(i)[2], dtype=torch.float32).to(device)
            predict_pkl = model_step1(query_dict.unsqueeze(0), query_dict_1d.unsqueeze(0))
            predict_pkl_dict[query_seqid] = predict_pkl

    for key, value in predict_pkl_dict.items():
        if hasattr(value, 'shape'):
            print(f"Key: {key}, Shape: {value.shape}")
        else:
            print(f"Key: {key} does not have a shape attribute")

    predict_pkl_dict_out_path = os.path.join(output_dir, 'predict_pkl_dict.pkl')
    with open(predict_pkl_dict_out_path, 'wb') as output_file:
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
    feature_data_file = os.path.join(output_dir, 'features_step2.pkl')
    with open(feature_data_file, 'wb') as output_file:
        pickle.dump(feature_data_dict, output_file)

    model_step2 = esm_ss_predict_tri_step2(embed_dim=128)
    checkpoint = torch.load(pre_model_name, map_location=lambda storage, loc: storage.cuda(0))
    model_step2.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_step2.to(device)

    ss_valid_dataset = esm_ss_dataset_step2(feature_data_file)
    print(ss_valid_dataset.__getitem__(0))
    print(ss_valid_dataset)

    model_step2.eval()
    with torch.no_grad():
        predict_score_dict = {}
        for i in range(ss_valid_dataset.__len__()):
            item = ss_valid_dataset[i]
            query_seqid = item[0]
            target_seqid = item[1]
            query_feature = item[2].to(device)
            target_feature = item[3].to(device)

            predict_score = model_step2(query_feature, target_feature)

            if query_seqid not in predict_score_dict:
                predict_score_dict[query_seqid] = {}

            predict_score_dict[query_seqid][target_seqid] = predict_score.cpu().numpy()  # 转为numpy，便于后续操作

    for query_seqid, targets in predict_score_dict.items():
        for target_seqid, prediction in targets.items():
            print(f"Query: {query_seqid}, Target: {target_seqid}, Prediction: {prediction}")

    output_file_path = os.path.join(output_dir, 'final_scores.csv')
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['QuerySeqID', 'TargetSeqID', 'Score'])

        for query_seqid, targets in predict_score_dict.items():
            for target_seqid, prediction in targets.items():
                writer.writerow([query_seqid, target_seqid, prediction.item()])  # 取出numpy的数值

    print(f"Predictions saved to {output_file_path}")

    return output_file_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, required=True, help="Path to the query FASTA file")
    parser.add_argument('--target_path', type=str, required=False, help="Path to the target FASTA file")
    parser.add_argument('--pairs_file', type=str, required=False, help="Path to the pairs CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output files")

    parser.add_argument('--esm_path', type=str, default=os.path.join(script_dir, 'esm', 'saved_models', 'ESM-2', 'esm2_t33_650M_UR50D.pt'), help="Path to the ESM model file")
    parser.add_argument('--embedding_script', type=str, default=os.path.join(script_dir, 'embedding_generate.py'), help="Path to the embedding generation script")

    args = parser.parse_args()

    if not(args.pairs_file or args.target_path):
        raise ValueError("If target_path is not provided, pairs_file must be provided.")

    query_path = process_msa_file(args.query_path)
    target_path = process_msa_file(args.target_path)
    results_file = predict_step1(query_path, target_path, args.pairs_file, args.output_dir, args.esm_path, args.embedding_script)
    print(f"TM-score results saved to: {results_file}")
