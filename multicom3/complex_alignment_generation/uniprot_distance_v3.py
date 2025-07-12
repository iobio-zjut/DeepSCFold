import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
from multicom3.monomer_alignment_generation.alignment import *
import os
import subprocess
import pickle
from MSArank_model.pss_for_msa_rank import predict_step1
from dataProcessingUtils import *

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

def create_species_dict(msa_df):
    species_lookup = {}
    for species, species_df in msa_df.groupby('prefix'):
        species_lookup[species] = species_df
    return species_lookup

def match_rows_by_sequence_similarity(this_species_msa_dfs,alignments):
    all_lengths_greater_than_2000 = all(2 < len(chain_alignment.headers) < 2000 for chain_alignment in alignments)
    if all_lengths_greater_than_2000:
        all_paired_msa_rows = []
        num_seqs = [len(species_df) for species_df in this_species_msa_dfs if species_df is not None]
        take_num_seqs = np.min(num_seqs)
        sort_by_similarity = (lambda x: x.sort_values('PLM_similarity', axis=0, ascending=False))
        for species_df in this_species_msa_dfs:
            if species_df is not None:
                species_df_sorted = sort_by_similarity(species_df)
                msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
            else:
                msa_rows = [-1] * take_num_seqs  # take the last 'padding' row
            all_paired_msa_rows.append(msa_rows)
        all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
        return all_paired_msa_rows
    else:
        all_paired_msa_rows = []
        num_seqs = [len(species_df) for species_df in this_species_msa_dfs if species_df is not None]
        take_num_seqs = np.min(num_seqs)
        sort_by_similarity = (lambda x: x.sort_values('msa_similarity', axis=0, ascending=False))
        for species_df in this_species_msa_dfs:
            if species_df is not None:
                species_df_sorted = sort_by_similarity(species_df)
                msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
            else:
                msa_rows = [-1] * take_num_seqs  # take the last 'padding' row
            all_paired_msa_rows.append(msa_rows)
        all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
        return all_paired_msa_rows



def reorder_paired_rows(all_paired_msa_rows_dict):
    all_paired_msa_rows = []

    for num_pairings in sorted(all_paired_msa_rows_dict, reverse=True):
        paired_rows = all_paired_msa_rows_dict[num_pairings]
        paired_rows_product = abs(np.array([np.prod(rows) for rows in paired_rows]))
        paired_rows_sort_index = np.argsort(paired_rows_product)
        all_paired_msa_rows.extend(paired_rows[paired_rows_sort_index])

    return np.array(all_paired_msa_rows)

def parse_header(header):
    # discard anything in header that might come after the
    # first whitespace (by convention this is typically annotation)
    header = header.split()[0]

    # try to extract region from sequence header
    m = re.search("(.+)/(\d+)-(\d+)", header)
    if m:
        id_, start_str, end_str = m.groups()
        region_start, region_end = int(start_str), int(end_str)
        return id_, region_start, region_end
    else:
        # cannot find region, so just give back sequence iD
        return header, None, None

def make_msa_df(alignment):
    regex = re.compile("^[A-Za-z0-9]+$")
    ids = []
    prefixes = []
    msa_similarity = []
    msa_row = []
    for i, id in enumerate(alignment.ids):
            
        if re.fullmatch(regex, id.split('_')[-1]) is None or id.split('_')[-1].startswith("UPI"):
            continue

        prefixes += [id[:-1]]
        ids += [id]
        per_seq_similarity = len([1 for j in range(len(alignment.main_seq)) if alignment.main_seq[j] == alignment.seqs[i][j]]) / float(len(alignment.main_seq))
        msa_similarity += [per_seq_similarity]
        msa_row += [i]

    return pd.DataFrame({'id': ids, 'prefix': prefixes, 'msa_similarity': msa_similarity, 'msa_row': msa_row})


def make_msa_PLMdf(alignment):
    regex = re.compile("^[A-Za-z0-9]+$")
    ids = []
    prefixes = []
    msa_similarity = []
    msa_row = []
    for i, id in enumerate(alignment.ids):

        if re.fullmatch(regex, id.split('_')[-1]) is None or id.split('_')[-1].startswith("UPI"):
            continue

        prefixes += [id[:-1]]
        ids += [id]
        per_seq_similarity = len(
            [1 for j in range(len(alignment.main_seq)) if alignment.main_seq[j] == alignment.seqs[i][j]]) / float(
            len(alignment.main_seq))
        msa_similarity += [per_seq_similarity]
        msa_row += [i]

    name = alignment.main_id
    query_fasta = f'{name}_query.fasta'
    multicom_a3m = f'{name}_multicom.a3m'

    with open(query_fasta, 'w') as queryfile:
        queryfile.write(">" + "query" + "\n")
        queryfile.write(alignment.main_seq)

    with open(multicom_a3m, 'w') as newa3m:
        for i in range(len(alignment.headers)):
            newa3m.write(">" + alignment.headers[i] + "\n")
            newa3m.write(alignment.seqs[i] + "\n")

    TM_score = predict_step1(query_fasta, multicom_a3m, name)

    PLM_similarity = []

    for i in range(len(msa_row)):
        PLM_similarity.append(TM_score[msa_row[i]])

    return pd.DataFrame({'id': ids, 'prefix': prefixes, 'msa_similarity': msa_similarity, 'msa_row': msa_row, 'PLM_similarity': PLM_similarity})


class UNIPROT_distance_v3:

    def get_interactions_v2(alignments, is_homomers=False):
        num_examples = len(alignments)

        all_chain_species_dict = []
        common_species = set()
        all_lengths_greater_than_2000 = all(2 < len(chain_alignment.headers) < 2000 for chain_alignment in alignments)
        for chain_alignment in alignments:
            if all_lengths_greater_than_2000:
                msa_df = make_msa_PLMdf(chain_alignment)
            else:
                msa_df = make_msa_df(chain_alignment)
            species_dict = create_species_dict(msa_df)
            all_chain_species_dict.append(species_dict)
            common_species.update(set(species_dict))

        common_species = sorted(common_species)

        all_paired_msa_rows = [np.zeros(num_examples, int)]
        all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
        all_paired_msa_rows_dict[num_examples] = [np.zeros(num_examples, int)]

        for species in common_species:
            if not species:
                continue
            this_species_msa_dfs = []
            species_dfs_present = 0
            for species_dict in all_chain_species_dict:
                if species in species_dict:
                    this_species_msa_dfs.append(species_dict[species])
                    species_dfs_present += 1
                else:
                    this_species_msa_dfs.append(None)

            # Skip species that are present in only one chain.
            if species_dfs_present <= 1:
                continue

            paired_msa_rows = match_rows_by_sequence_similarity(this_species_msa_dfs,alignments)
            all_paired_msa_rows.extend(paired_msa_rows)
            all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)

        all_paired_msa_rows_dict = {
            num_examples: np.array(paired_msa_rows) for
            num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
        }
        
        paired_rows = reorder_paired_rows(all_paired_msa_rows_dict)

        # make sure each id only apprears once in the concatenated alignment
        seen_ids = {k: [] for k in range(num_examples)}
        paired_rows_filtered = []
        for pair_index in range(len(paired_rows[:, 0])):
            if pair_index == 0:
                paired_rows_filtered += [paired_rows[pair_index, :]]
                continue
        
            row_indices = list(paired_rows[pair_index, :])
            pass_filter = True
            for j in range(len(alignments)):
                index = row_indices[j]
                if index == -1:
                    continue
                else:
                    header, start, end = parse_header(alignments[j].headers[index])
                    if f"{header}_{start}-{end}" in seen_ids[j]:
                        pass_filter = False
                        break
            
            if pass_filter:
                paired_rows_filtered += [paired_rows[pair_index]]
                for j in range(len(alignments)):
                    index = row_indices[j]
                    if index == -1:
                        continue
                    else:
                        header, start, end = parse_header(alignments[j].headers[index])
                        seen_ids[j] += [f"{header}_{start}-{end}"]

        return np.array(paired_rows_filtered)