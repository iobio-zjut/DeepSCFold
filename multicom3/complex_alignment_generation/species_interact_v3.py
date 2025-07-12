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

def read_species_annotation_table(annotation):

    SPECIES_ANNOTATION_COLUMNS = ["OS", "Tax"]

    data = annotation

    # initialize the column to extract the species information from
    annotation_column = None
    current_num_annotations = 0

    # Determine whether to extract based on the "OS" field
    # or the "Tax" field. Generally, OS is for Uniprot
    for column in SPECIES_ANNOTATION_COLUMNS:
        # if this column contains more non-null values
        if column not in data:
            continue

        num_annotations = sum(data[column].notnull())
        if num_annotations > current_num_annotations:
            # use that column to extract data
            annotation_column = column
            current_num_annotations = num_annotations

    # if we did not find an annotation column, return an error
    if annotation_column is None:
        return None

    # creates a new column called species with the species annotations
    data.loc[:, "species"] = data.loc[:, annotation_column]

    return data[["id", "name", "species"]]

def extract_header_annotation(alignment):
    columns = [
            ("GN", "gene"),
            ("OS", "organism"),
            ("PE", "existence_evidence"),
            ("SV", "sequence_version"),
            ("n", "num_cluster_members"),
            ("Tax", "taxon"),
            ("RepID", "representative_member")
        ]

    col_to_descr = OrderedDict(columns)
    regex = re.compile("\s({})=".format("|".join(col_to_descr.keys())))

    res = []
    for seq_idx, seq_id in enumerate(alignment.ids):
        full_header = alignment.headers[seq_idx]
        anno = None
        if ("GS" in alignment.annotation and
                    seq_id in alignment.annotation["GS"] and
                    "DE" in alignment.annotation["GS"][seq_id]):
            anno = alignment.annotation["GS"][seq_id]["DE"]
        else:
            split = full_header.split(maxsplit=1)
            if len(split) == 2:
                _, anno = split

        # extract info from line if we got one
        if anno is not None:
            # do split on known field names o keep things
            # simpler than a gigantic full regex to match
            # (some fields are allowed to be missing)
            pairs = re.split(regex, anno)
            pairs = ["id", seq_id, "name"] + pairs

            # create feature-value map
            feat_map = dict(zip(pairs[::2], pairs[1::2]))
            res.append(feat_map)
        else:
            res.append({"id": seq_id})

    df = pd.DataFrame(res)
    return df.reindex(
            ["id", "name"] + list(col_to_descr.keys()),
            axis=1
        )

def create_species_dict(msa_df):
    species_lookup = {}
    for species, species_df in msa_df.groupby('species'):
        species_lookup[species] = species_df
    return species_lookup

def match_rows_by_sequence_similarity(this_species_msa_dfs,alignments):
    all_lengths_between_2_and_2000 = all(2 < len(chain_alignment.headers) < 2000 for chain_alignment in alignments)
    if all_lengths_between_2_and_2000:
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


def make_msa_df(alignment):
    annotation = extract_header_annotation(alignment)

    annotation_table = read_species_annotation_table(annotation)

    ids = []
    species = []
    msa_similarity = []
    msa_row = []

    if annotation_table is None:
        return pd.DataFrame({'id': ids, 'species': species, 'msa_similarity': msa_similarity, 'msa_row': msa_row})

    for index, row in annotation_table.iterrows():
        ids += [row.id]
        species += [row.species]
        per_seq_similarity = len(
            [1 for j in range(len(alignment.main_seq)) if alignment.main_seq[j] == alignment.seqs[index][j]]) / float(
            len(alignment.main_seq))
        msa_similarity += [per_seq_similarity]
        msa_row += [index]

    return pd.DataFrame({'id': ids, 'species': species, 'msa_similarity': msa_similarity, 'msa_row': msa_row})

def make_msa_PLMdf(alignment):
    print(alignment)
    annotation = extract_header_annotation(alignment)
    annotation_table = read_species_annotation_table(annotation)
        
    ids = []
    species = []
    msa_similarity = []
    msa_row = []

    PLM_similarity = []

    if annotation_table is None:
        return pd.DataFrame({'id': ids, 'species': species, 'msa_similarity': msa_similarity, 'msa_row': msa_row, 'PLM_similarity': PLM_similarity})

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

    PLM_similarity = predict_step1(query_fasta, multicom_a3m, name)

    for index, row in annotation_table.iterrows():
        ids += [row.id]
        species += [row.species]
        per_seq_similarity = len([1 for j in range(len(alignment.main_seq)) if alignment.main_seq[j] == alignment.seqs[index][j]]) / float(len(alignment.main_seq))
        msa_similarity += [per_seq_similarity]
        msa_row += [index]

    # os.remove('query.fasta')
    # os.remove('multicom.a3m')

    return pd.DataFrame({'id': ids, 'species': species, 'msa_similarity': msa_similarity, 'msa_row': msa_row, 'PLM_similarity': PLM_similarity})

class Species_interact_v3:

    def get_interactions_v2(alignments):
        num_examples = len(alignments)

        all_chain_species_dict = []
        common_species = set()
        all_lengths_between_2_and_2000 = all(2 < len(chain_alignment.headers) < 2000 for chain_alignment in alignments)
        for chain_alignment in alignments:
            if all_lengths_between_2_and_2000:
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
        return paired_rows

