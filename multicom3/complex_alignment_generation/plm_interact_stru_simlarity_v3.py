import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
from multicom3.monomer_alignment_generation.alignment import *
import os
import subprocess
import pickle
from dataProcessingUtils import *
import glob

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")


def clean_folder(folder_path, keep_files):
    if not isinstance(keep_files, list):
        print("Error: keep_files should be a list of filenames to keep.")
        return

    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name not in keep_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
                print(f"Deleted empty folder: {dir_path}")
            except OSError as e:
                print(f"Folder not empty or failed to delete: {dir_path}")


def plm_interact_structure(outdir, alignment, a3m_alignment, file_name):
    dir = os.path.join(outdir, file_name)
    ensure_directory_exists(dir)

    temp_chainA_fasta = os.path.join(dir, 'chainA.fasta')
    temp_chainA_a3m = os.path.join(dir, 'chainA.a3m')
    temp_chainB_fasta = os.path.join(dir, 'chainB.fasta')
    temp_chainB_a3m = os.path.join(dir, 'chainB.a3m')
    temp_chainC_fasta = os.path.join(dir, 'chainC.fasta')
    temp_chainC_a3m = os.path.join(dir, 'chainC.a3m')

    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    prepro = os.path.join(project_root, "MSArank_model", "predict_for_msa_rank_v2.py")

    temp_csv_input = os.path.join(dir, 'final_scores.csv')
    temp_csv_output = os.path.join(dir, 'sort_scores.csv')
    deala3m = os.path.join(project_root, "MSArank_model", "msa_deal_for_MULTICOM.py")

    temp_chainA_a3m_ranked = os.path.join(dir, 'chainA_ranked.a3m')
    temp_chainB_a3m_ranked = os.path.join(dir, 'chainB_ranked.a3m')
    temp_chainC_a3m_ranked = os.path.join(dir, 'chainC_ranked.a3m')
    code_dimer = os.path.join(project_root, "seq_interact_model", "generation_for_dimer.py")
    code_trimer = os.path.join(project_root, "seq_interact_model", "generation_for_trimer.py")
    con_A = alignment['chain1']['name'] + '_con.a3m'
    con_B = alignment['chain2']['name'] + '_con.a3m'

    # monomer MSA rank by stru_sim
    if len(a3m_alignment) > 0:
        print("uniref_a3m_alignments")

        if len(a3m_alignment) == 2:
            # chainA
            with open(temp_chainA_fasta, 'w') as file:
                file.write(
                    '>' + str(a3m_alignment[0].main_id) + '\n' + str(a3m_alignment[0].main_seq) + '\n')
            with open(temp_chainA_a3m, 'w') as file:
                for i in range(len(a3m_alignment[0].seqs)):
                    file.write(
                        '>' + a3m_alignment[0].headers[i] + '\n' + a3m_alignment[0].seqs[i] + '\n')
            os.system(
                f"python {prepro} --query {temp_chainA_fasta} --target {temp_chainA_a3m} --output {dir}"
            )

            os.system(
                f"python {deala3m} --input_csv_file {temp_csv_input} --output_csv_file  {temp_csv_output} --input_msa_file  {temp_chainA_a3m} --output_sorted_msa_file {temp_chainA_a3m_ranked} --input_fasta_file {temp_chainA_fasta}")

            # chainB
            with open(temp_chainB_fasta, 'w') as file:
                file.write(
                    '>' + str(a3m_alignment[1].main_id) + '\n' + str(a3m_alignment[1].main_seq) + '\n')
            with open(temp_chainB_a3m, 'w') as file:
                for i in range(len(a3m_alignment[1].seqs)):
                    file.write(
                        '>' + a3m_alignment[1].headers[i] + '\n' + a3m_alignment[1].seqs[i] + '\n')
            os.system(
                f"python {prepro} --query {temp_chainB_fasta} --target {temp_chainB_a3m} --output {dir}"
            )

            os.system(
                f"python {deala3m} --input_csv_file {temp_csv_input} --output_csv_file  {temp_csv_output} --input_msa_file  {temp_chainB_a3m} --output_sorted_msa_file {temp_chainB_a3m_ranked} --input_fasta_file {temp_chainB_fasta}")

        else:
            # chainA
            with open(temp_chainA_fasta, 'w') as file:
                file.write(
                    '>' + str(a3m_alignment[0].main_id) + '\n' + str(a3m_alignment[0].main_seq) + '\n')
            with open(temp_chainA_a3m, 'w') as file:
                for i in range(len(a3m_alignment[0].seqs)):
                    file.write(
                        '>' + a3m_alignment[0].headers[i] + '\n' + a3m_alignment[0].seqs[i] + '\n')
            os.system(
                f"python {prepro} --query {temp_chainA_fasta} --target {temp_chainA_a3m} --output {dir}"
            )

            os.system(
                f"python {deala3m} --input_csv_file {temp_csv_input} --output_csv_file  {temp_csv_output} --input_msa_file  {temp_chainA_a3m} --output_sorted_msa_file {temp_chainA_a3m_ranked} --input_fasta_file {temp_chainA_fasta}")

            # chainB
            with open(temp_chainB_fasta, 'w') as file:
                file.write(
                    '>' + str(a3m_alignment[1].main_id) + '\n' + str(a3m_alignment[1].main_seq) + '\n')
            with open(temp_chainB_a3m, 'w') as file:
                for i in range(len(a3m_alignment[1].seqs)):
                    file.write(
                        '>' + a3m_alignment[1].headers[i] + '\n' + a3m_alignment[1].seqs[i] + '\n')
            os.system(
                f"python {prepro} --query {temp_chainB_fasta} --target {temp_chainB_a3m} --output {dir}"
            )

            os.system(
                f"python {deala3m} --input_csv_file {temp_csv_input} --output_csv_file  {temp_csv_output} --input_msa_file  {temp_chainB_a3m} --output_sorted_msa_file {temp_chainB_a3m_ranked} --input_fasta_file {temp_chainB_fasta}")

            # chainC
            with open(temp_chainC_fasta, 'w') as file:
                file.write(
                    '>' + str(a3m_alignment[2].main_id) + '\n' + str(a3m_alignment[2].main_seq) + '\n')
            with open(temp_chainC_a3m, 'w') as file:
                for i in range(len(a3m_alignment[2].seqs)):
                    file.write(
                        '>' + a3m_alignment[2].headers[i] + '\n' + a3m_alignment[2].seqs[i] + '\n')
            os.system(
                f"python {prepro} --query {temp_chainC_fasta} --target {temp_chainC_a3m} --output {dir}"
            )

            os.system(
                f"python {deala3m} --input_csv_file {temp_csv_input} --output_csv_file  {temp_csv_output} --input_msa_file  {temp_chainC_a3m} --output_sorted_msa_file {temp_chainC_a3m_ranked} --input_fasta_file {temp_chainC_fasta}")

    # pair-MSA generation
    if len(a3m_alignment) > 0:
        print("uniref_a3m_alignments")

        complex_a3m = file_name + '.a3m'
        out_csv = file_name + '_interact.csv'

        if len(a3m_alignment) == 2:
            os.system(
                f"python {code_dimer} --input_msa_1 {temp_chainA_a3m_ranked} --input_msa_2 {temp_chainB_a3m} --output_dir {dir} --output_pairs_csv {out_csv} --output_msa_for_chain1 {con_A} --output_msa_for_chain2 {con_B} --output_complex_msa {complex_a3m}"
            )

            keep_files = [out_csv, con_A, con_B, complex_a3m]  # 要保留的文件列表
            print(keep_files)
            clean_folder(dir, keep_files)

        else:
            con_C = alignment['chain3']['name'] + '_con.a3m'
            os.system(
                f"python {code_trimer} --input_msa_1 {temp_chainA_a3m_ranked} --input_msa_2 {temp_chainB_a3m_ranked} --input_msa_3 {temp_chainC_a3m_ranked} --output_dir {dir} --output_pairs_csv_fileName {out_csv} --output_msa_for_chain1 {con_A} --output_msa_for_chain2 {con_B} --output_msa_for_chain3 {con_C} --output_complex_msa {complex_a3m}"
            )

            keep_files = [out_csv, con_A, con_B, con_C, complex_a3m]  # 要保留的文件列表
            print(keep_files)
            clean_folder(dir, keep_files)
