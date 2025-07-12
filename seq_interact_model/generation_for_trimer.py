import os
import argparse
import subprocess
import pandas as pd
import csv


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

def generation_for_dimer(script_path, input_msaName_1, input_msaName_2, Path_to_outputdir,
                         output_pairs_csv_fileName, output_msa_for_chain1, output_msa_for_chain2,
                         output_complex_msa):
    command = f"python {script_path} --input_msa_1 {input_msaName_1} --input_msa_2 {input_msaName_2} --output_dir {Path_to_outputdir} --output_pairs_csv {output_pairs_csv_fileName} --output_msa_for_chain1 {output_msa_for_chain1} --output_msa_for_chain2 {output_msa_for_chain2} --output_complex_msa {output_complex_msa}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True, check=True)


def extract_sequences_from_a3m(file_path):
    sequences = {}
    current_header = None
    current_sequence = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                current_header = line
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_header:
            sequences[current_header] = ''.join(current_sequence)

    return sequences


def find_common_sequences(seqs_AB, seqs_AC):
    common_headers = set(seqs_AB.keys()) & set(seqs_AC.keys())
    common_headers_in_order = [header for header in seqs_AB.keys() if header in common_headers]
    return common_headers_in_order

def write_fasta(sequences, output_file, order):
    with open(output_file, 'w') as f:
        for header in order:
            if header in sequences:
                f.write(f"{header}\n")
                f.write(f"{sequences[header]}\n")

def write_fasta_v2(sequences, output_file, order):
    with open(output_file, 'w') as f:
        for header in order:
            header_with_gt = f">{header}"
            if header_with_gt in sequences:
                f.write(f"{header_with_gt}\n")
                f.write(f"{sequences[header_with_gt]}\n")

def extract_matching_sequences(common_headers_order, seq_file):
    all_sequences = extract_sequences_from_a3m(seq_file)

    all_sequences_no_gt = {header.lstrip('>'): seq for header, seq in all_sequences.items()}

    matching_sequences = {f">{header}": all_sequences_no_gt[header] for header in common_headers_order if
                          header in all_sequences_no_gt}

    return matching_sequences

def write_combined_msa(chainA_seqs, chainB_seqs, chainC_seqs, output_file):
    with open(output_file, 'w') as f:
        for header in chainA_seqs.keys():
            f.write(f"{header}_____{header}\n")
            f.write(f"{chainA_seqs[header]}{chainB_seqs[header]}{chainC_seqs[header]}\n")


def extract_pairs_from_csv(csv_file_path, common_headers_order):
    pair_chain_headers = []
    common_headers_set = set(header.lstrip('>') for header in common_headers_order)

    pairs_dict = {}

    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if row[0] in common_headers_set:
                pairs_dict[row[0]] = row[2]

    for header in common_headers_order:
        header_stripped = header.lstrip('>')
        if header_stripped in pairs_dict:
            pair_chain_headers.append(pairs_dict[header_stripped])

    return pair_chain_headers


def generate_output_pairs_csv(output_csv_path, chainA_headers, chainB_headers, chainC_headers):
    assert len(chainA_headers) == len(chainB_headers) == len(chainC_headers), "Headers length mismatch!"

    data = []
    for idx, (headerA, headerB, headerC) in enumerate(zip(chainA_headers, chainB_headers, chainC_headers)):
        data.append([headerA.lstrip('>'), idx, headerB, idx, headerC, idx])

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id_1', 'index_1', 'id_2', 'index_2', 'id_3', 'index_3'])
        csv_writer.writerows(data)

def write_complex_msa(output_file, chainA_headers, chainB_headers, chainC_headers, chainA_seqs, chainB_seqs, chainC_seqs):
    assert len(chainA_headers) == len(chainB_headers) == len(chainC_headers), "Headers length mismatch!"
    assert len(chainA_seqs) == len(chainB_seqs) == len(chainC_seqs), "Sequences length mismatch!"

    with open(output_file, 'w') as f:
        for headerA, headerB, headerC in zip(chainA_headers, chainB_headers, chainC_headers):
            combined_header = f"{headerA}_____{headerB}_____{headerC}"
            combined_sequence = f"{chainA_seqs[f'{headerA}']}{chainB_seqs[f'>{headerB}']}{chainC_seqs[f'>{headerC}']}"
            f.write(f"{combined_header}\n")
            f.write(f"{combined_sequence}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and split sequences from input FASTA files.')
    # input
    parser.add_argument('--input_msa_1', type=str, required=True, help='Path to the first input MSA file.')
    parser.add_argument('--input_msa_2', type=str, required=True, help='Path to the second input MSA file.')
    parser.add_argument('--input_msa_3', type=str, required=True, help='Path to the third input MSA file.')

    # output
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--output_pairs_csv_fileName', type=str, required=False, default='output_pairs.csv',
                        help='The name to hold the output CSV file. Default is "output_pairs.csv".')
    # species_interact_uniprot_sto_interact.csv
    parser.add_argument('--output_msa_for_chain1', type=str, required=False, default='ChainA_con.a3m',
                        help='The name to hold the MSA file extracted sequences from the first input file. Default is "ChainA_con.a3m".')
    parser.add_argument('--output_msa_for_chain2', type=str, required=False, default='ChainB_con.a3m',
                        help='The name to hold the MSA file extracted sequences from the second input file. Default is "ChainB_con.a3m".')
    parser.add_argument('--output_msa_for_chain3', type=str, required=False, default='ChainC_con.a3m',
                        help='The name to hold the MSA file extracted sequences from the third input file. Default is "ChainC_con.a3m".')
    parser.add_argument('--output_complex_msa', type=str, required=False, default='output_complex_msa.a3m',
                        help='File name of the output pair-msa. Default is "output_complex_msa.a3m".')

    args = parser.parse_args()
    ensure_directory_exists(args.output_dir)

    # dimer generation
    dimer_msa_script_path = "/path/to/generation_for_dimer.py"

    Path_to_outputdir_AB = os.path.join(args.output_dir, "chain_AB")
    generation_for_dimer(dimer_msa_script_path, args.input_msa_1, args.input_msa_2, Path_to_outputdir_AB,
                         args.output_pairs_csv_fileName, args.output_msa_for_chain1, args.output_msa_for_chain2,
                         args.output_complex_msa)

    Path_to_outputdir_AC = os.path.join(args.output_dir, "chain_AC")
    generation_for_dimer(dimer_msa_script_path, args.input_msa_1, args.input_msa_3, Path_to_outputdir_AC,
                         args.output_pairs_csv_fileName, args.output_msa_for_chain1, args.output_msa_for_chain3,
                         args.output_complex_msa)

    #trimer generation
    seqs_chain_AB_A = extract_sequences_from_a3m(os.path.join(Path_to_outputdir_AB, args.output_msa_for_chain1))
    seqs_chain_AC_A = extract_sequences_from_a3m(os.path.join(Path_to_outputdir_AC, args.output_msa_for_chain1))

    common_headers_order = find_common_sequences(seqs_chain_AB_A, seqs_chain_AC_A)
    print('chainA_headers:', len(common_headers_order))
    common_sequences_A = {header: seqs_chain_AB_A[header] for header in common_headers_order}

    output_chainA = os.path.join(args.output_dir, args.output_msa_for_chain1)
    write_fasta(common_sequences_A, output_chainA, common_headers_order)

    pairs_csv_file_path_AB = os.path.join(Path_to_outputdir_AB, args.output_pairs_csv_fileName)
    chainB_headers = extract_pairs_from_csv(pairs_csv_file_path_AB, common_headers_order)
    print('chainB_headers:', len(chainB_headers))
    pairs_csv_file_path_AC = os.path.join(Path_to_outputdir_AC, args.output_pairs_csv_fileName)
    chainC_headers = extract_pairs_from_csv(pairs_csv_file_path_AC, common_headers_order)
    print('chainC_headers:', len(chainC_headers))

    seqs_chain_B = extract_matching_sequences(chainB_headers,
                                              os.path.join(Path_to_outputdir_AB, args.output_msa_for_chain2))
    output_chainB = os.path.join(args.output_dir, args.output_msa_for_chain2)
    write_fasta_v2(seqs_chain_B, output_chainB, chainB_headers)

    seqs_chain_C = extract_matching_sequences(chainC_headers,
                                              os.path.join(Path_to_outputdir_AC, args.output_msa_for_chain3))
    output_chainC = os.path.join(args.output_dir, args.output_msa_for_chain3)
    write_fasta_v2(seqs_chain_C, output_chainC, chainC_headers)

    output_pairs_csv_path = os.path.join(args.output_dir, args.output_pairs_csv_fileName)
    generate_output_pairs_csv(output_pairs_csv_path, common_headers_order, chainB_headers, chainC_headers)

    output_complex_msa_path = os.path.join(args.output_dir, args.output_complex_msa)
    write_complex_msa(output_complex_msa_path, common_headers_order, chainB_headers, chainC_headers, common_sequences_A,
                      seqs_chain_B, seqs_chain_C)