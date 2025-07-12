import os
import argparse
import subprocess
import pandas as pd
import csv


def generation_for_dimer(script_path, input_msa1, input_msa2, output_dir, output_pairs_csv, output_msa1, output_msa2):
    command = f"python {script_path} --input_msa_1 {input_msa1} --input_msa_2 {input_msa2} --output_dir {output_dir} --output_pairs_csv {output_pairs_csv} --output_msa_for_chain1 {output_msa1} --output_msa_for_chain2 {output_msa2}"
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


def find_common_sequences(list_of_sequence_dicts):
    common_headers = set.intersection(*[set(seq_dict.keys()) for seq_dict in list_of_sequence_dicts])
    common_headers_in_order = [header for header in list_of_sequence_dicts[0].keys() if header in common_headers]
    return common_headers_in_order


def write_fasta(sequences, output_file, order):
    with open(output_file, 'w') as f:
        for header in order:
            if header in sequences:
                f.write(f"{header}\n")
                f.write(f"{sequences[header]}\n")


def extract_matching_sequences(common_headers_order, seq_file):
    all_sequences = extract_sequences_from_a3m(seq_file)
    all_sequences_no_gt = {header.lstrip('>'): seq for header, seq in all_sequences.items()}
    matching_sequences = {f">{header}": all_sequences_no_gt[header] for header in common_headers_order if
                          header in all_sequences_no_gt}
    return matching_sequences


def write_complex_msa(output_file, chain_headers, chain_sequences):
    chain_count = len(chain_headers)
    with open(output_file, 'w') as f:
        for i in range(len(chain_headers[0])):
            combined_header = "_____".join([chain_headers[chain_idx][i] for chain_idx in range(chain_count)])
            combined_sequence = "".join(
                [chain_sequences[chain_idx][f'>{chain_headers[chain_idx][i]}'] for chain_idx in range(chain_count)])
            f.write(f"{combined_header}\n")
            f.write(f"{combined_sequence}\n")


def generate_output_pairs_csv(output_csv_path, chain_headers):
    chain_count = len(chain_headers)
    assert all(len(chain_headers[0]) == len(headers) for headers in chain_headers), "Headers length mismatch!"

    data = []
    for idx in range(len(chain_headers[0])):
        row = []
        for chain_idx in range(chain_count):
            row.extend([chain_headers[chain_idx][idx].lstrip('>'), idx])
        data.append(row)

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = []
        for chain_idx in range(chain_count):
            headers.extend([f'id_{chain_idx + 1}', f'index_{chain_idx + 1}'])
        csv_writer.writerow(headers)
        csv_writer.writerows(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process sequences from input MSA files.')
    parser.add_argument('--input_msa', type=str, nargs='+', required=True, help='Paths to the input MSA files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--output_pairs_csv', type=str, required=False, default='output_pairs.csv',
                        help='Filename for output CSV with chain pairs. Default is "output_pairs.csv".')
    parser.add_argument('--output_complex_msa', type=str, required=False, default='output_complex_msa.a3m',
                        help='Filename for output complex MSA. Default is "output_complex_msa.a3m".')

    args = parser.parse_args()

    input_msa_files = args.input_msa
    chain_count = len(input_msa_files)
    output_dir = args.output_dir

    dimer_msa_script_path = "/path/to/generation_for_dimer.py"
    pair_output_dirs = []
    for i in range(chain_count - 1):
        output_dir_i = os.path.join(output_dir, f"chain_{i + 1}_{i + 2}")
        pair_output_dirs.append(output_dir_i)
        generation_for_dimer(dimer_msa_script_path, input_msa_files[i], input_msa_files[i + 1], output_dir_i,
                             args.output_pairs_csv, f'Chain{i + 1}_con.a3m', f'Chain{i + 2}_con.a3m')

    chain_sequences = []
    for i, pair_dir in enumerate(pair_output_dirs):
        seqs_chain = extract_sequences_from_a3m(os.path.join(pair_dir, f'Chain{i + 1}_con.a3m'))
        chain_sequences.append(seqs_chain)

    common_headers_order = find_common_sequences(chain_sequences)

    final_chain_headers = []
    final_chain_sequences = []
    for i, pair_dir in enumerate(pair_output_dirs):
        matching_seqs = extract_matching_sequences(common_headers_order,
                                                   os.path.join(pair_dir, f'Chain{i + 1}_con.a3m'))
        final_chain_headers.append([header.lstrip('>') for header in common_headers_order])
        final_chain_sequences.append(matching_seqs)

    output_complex_msa_path = os.path.join(output_dir, args.output_complex_msa)
    write_complex_msa(output_complex_msa_path, final_chain_headers, final_chain_sequences)

    output_pairs_csv_path = os.path.join(output_dir, args.output_pairs_csv)
    generate_output_pairs_csv(output_pairs_csv_path, final_chain_headers)
